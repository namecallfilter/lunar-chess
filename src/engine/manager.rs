use std::{
	collections::VecDeque,
	io::Write,
	sync::{
		Arc, Mutex,
		atomic::{AtomicBool, Ordering},
		mpsc::{Receiver, Sender, channel},
	},
	thread::{self, JoinHandle},
};

use anyhow::Result;

use crate::{
	chess::PlayerColor,
	engine::{EngineWrapper, MoveWithScore},
	errors::Fen,
};

pub enum EngineCommand {
	SetPosition(Fen, Option<PlayerColor>),
	StartSearch,
	StopSearch,
	Quit,
	InternalSearchComplete,
}

pub struct EngineManager {
	command_tx: Sender<EngineCommand>,
	thread_handle: Option<JoinHandle<()>>,
}

impl EngineManager {
	pub fn new<F>(on_update: F) -> Result<Self>
	where
		F: Fn(Vec<MoveWithScore>, Option<PlayerColor>) + Send + Sync + 'static,
	{
		let (command_tx, command_rx) = channel();
		let tx_for_loop = command_tx.clone();

		let thread_handle = thread::spawn(move || {
			Self::run_loop(command_rx, tx_for_loop, on_update);
		});

		Ok(Self {
			command_tx,
			thread_handle: Some(thread_handle),
		})
	}

	fn run_loop<F>(
		command_rx: Receiver<EngineCommand>, command_tx: Sender<EngineCommand>, on_update: F,
	) where
		F: Fn(Vec<MoveWithScore>, Option<PlayerColor>) + Send + Sync + 'static,
	{
		let wrapper = match EngineWrapper::new() {
			Ok(w) => w,
			Err(e) => {
				tracing::error!("Failed to start engine: {}", e);
				return;
			}
		};

		let stopper = wrapper.get_stopper();
		let wrapper = Arc::new(Mutex::new(wrapper));
		let on_update = Arc::new(on_update);
		let is_searching = Arc::new(AtomicBool::new(false));

		let mut search_handle: Option<JoinHandle<()>> = None;
		let mut current_color: Option<PlayerColor> = None;
		let mut pending_commands: VecDeque<EngineCommand> = VecDeque::new();

		loop {
			let command = if !pending_commands.is_empty() && search_handle.is_none() {
				pending_commands.pop_front()
			} else {
				command_rx.recv().ok()
			};

			let command = match command {
				Some(c) => c,
				None => break,
			};

			match command {
				EngineCommand::StopSearch => {
					if is_searching.load(Ordering::Acquire) {
						let _ = stopper.clone().write(b"stop\n");
					}
				}
				EngineCommand::SetPosition(fen, color) => {
					if search_handle.is_some() {
						if is_searching.load(Ordering::Acquire) {
							let _ = stopper.clone().write(b"stop\n");
						}

						pending_commands.push_back(EngineCommand::SetPosition(fen, color));
					} else {
						current_color = color;

						let mut guard = match wrapper.lock() {
							Ok(g) => g,
							Err(poisoned) => {
								tracing::error!("Engine mutex poisoned; recovering.");
								poisoned.into_inner()
							}
						};

						if let Err(e) = guard.set_position(&fen) {
							tracing::error!("Failed to set position: {}", e);
						}
					}
				}
				EngineCommand::StartSearch => {
					if search_handle.is_some() {
						if is_searching.load(Ordering::Acquire) {
							let _ = stopper.clone().write(b"stop\n");
						}

						pending_commands.push_back(EngineCommand::StartSearch);
					} else {
						let wrapper = wrapper.clone();
						let on_update = on_update.clone();
						let color = current_color;
						let is_searching = is_searching.clone();
						let tx = command_tx.clone();

						search_handle = Some(thread::spawn(move || {
							is_searching.store(true, Ordering::Release);

							let mut guard = match wrapper.lock() {
								Ok(g) => g,
								Err(poisoned) => {
									tracing::error!(
										"Engine mutex poisoned in search thread; recovering."
									);
									poisoned.into_inner()
								}
							};

							let result = guard.get_best_moves(|moves| {
								on_update(moves.to_vec(), color);
							});

							is_searching.store(false, Ordering::Release);

							if let Err(e) = result {
								tracing::error!("Search failed: {}", e);
							}

							let _ = tx.send(EngineCommand::InternalSearchComplete);
						}));
					}
				}
				EngineCommand::InternalSearchComplete => {
					if let Some(handle) = search_handle.take() {
						let _ = handle.join();
					}
				}
				EngineCommand::Quit => {
					if search_handle.is_some() {
						if is_searching.load(Ordering::Acquire) {
							let _ = stopper.clone().write(b"stop\n");
						}

						pending_commands.push_back(EngineCommand::Quit);
					} else {
						break;
					}
				}
			}
		}
	}

	pub fn send(&self, command: EngineCommand) {
		if let Err(e) = self.command_tx.send(command) {
			tracing::warn!("Failed to send EngineCommand: {}", e);
		}
	}
}

impl Drop for EngineManager {
	fn drop(&mut self) {
		let _ = self.command_tx.send(EngineCommand::Quit);
		if let Some(handle) = self.thread_handle.take() {
			let _ = handle.join();
		}
	}
}
