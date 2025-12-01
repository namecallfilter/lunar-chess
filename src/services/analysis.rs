use std::{
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
	},
	thread::{self, JoinHandle},
	time::Duration,
};

use parking_lot::Condvar;

use crate::{
	chess::{ChessMove, board},
	engine::EngineWrapper,
	errors::Fen,
	ui::{SharedBoardState, UserEvent},
};

const SHUTDOWN_CHECK_INTERVAL: Duration = Duration::from_secs(1);
const ENGINE_RESTART_DELAY: Duration = Duration::from_secs(1);
const ENGINE_INIT_RETRY_DELAY: Duration = Duration::from_secs(5);

pub struct AnalysisService {
	thread_handle: Option<JoinHandle<()>>,
	shutdown: Arc<AtomicBool>,
}

impl AnalysisService {
	pub fn spawn(
		proxy: winit::event_loop::EventLoopProxy<UserEvent>, board_state: SharedBoardState,
		board_changed: Arc<Condvar>,
	) -> Self {
		let shutdown = Arc::new(AtomicBool::new(false));
		let shutdown_clone = Arc::clone(&shutdown);

		let handle = thread::spawn(move || {
			run_analysis_loop(proxy, board_state, board_changed, shutdown_clone);
		});

		Self {
			thread_handle: Some(handle),
			shutdown,
		}
	}

	fn shutdown(&self) {
		self.shutdown.store(true, Ordering::SeqCst);
	}
}

impl Drop for AnalysisService {
	fn drop(&mut self) {
		self.shutdown();
		if let Some(handle) = self.thread_handle.take() {
			let _ = handle.join();
		}
	}
}

fn run_analysis_loop(
	proxy: winit::event_loop::EventLoopProxy<UserEvent>, board_state: SharedBoardState,
	board_changed: Arc<Condvar>, shutdown: Arc<AtomicBool>,
) {
	let mut last_analyzed_fen: Option<Fen> = None;

	while !shutdown.load(Ordering::SeqCst) {
		let engine_result = EngineWrapper::new();

		match engine_result {
			Ok(mut engine) => {
				tracing::info!("Analysis engine ready");

				while !shutdown.load(Ordering::SeqCst) {
					let current_position = {
						let mut lock = board_state.lock();
						let _ = board_changed.wait_for(&mut lock, SHUTDOWN_CHECK_INTERVAL);
						lock.clone()
					};

					if shutdown.load(Ordering::SeqCst) {
						break;
					}

					if let Some(state) = current_position {
						let fen = match state.to_fen() {
							Ok(f) => f,
							Err(e) => {
								tracing::debug!("Cannot generate FEN: {}", e);
								continue;
							}
						};

						if last_analyzed_fen.as_ref() == Some(&fen) {
							continue;
						}

						proxy
							.send_event(UserEvent::UpdateBestMoves(Vec::new()))
							.ok();

						let player_color = match state.board.player_color() {
							Some(c) => c,
							None => {
								tracing::debug!("Board orientation unknown, skipping analysis");
								continue;
							}
						};

						match engine.set_position(&fen) {
							Ok(_) => {}
							Err(e) => {
								tracing::warn!("Engine error on position '{}': {:?}", fen, e);
								tracing::debug!("Engine may have crashed, restarting...");
								break;
							}
						}

						match engine.get_best_moves(|moves_with_scores| {
							if !moves_with_scores.is_empty() {
								let best_moves: Vec<ChessMove> = moves_with_scores
									.iter()
									.filter_map(|m| {
										board::parse_move(m.notation.as_str(), player_color).map(
											|mv| ChessMove::with_score(mv.from, mv.to, m.score),
										)
									})
									.collect();

								if !best_moves.is_empty() {
									proxy
										.send_event(UserEvent::UpdateBestMoves(best_moves))
										.ok();
								}
							}
						}) {
							Ok(_) => {
								last_analyzed_fen = Some(fen);
							}
							Err(e) => {
								tracing::warn!("Failed to get best moves: {:?}", e);
								tracing::debug!("Engine may have crashed, restarting...");
								break;
							}
						}
					}
				}

				if !shutdown.load(Ordering::SeqCst) {
					tracing::warn!("Engine crashed, restarting...");

					thread::sleep(ENGINE_RESTART_DELAY);
				}
			}
			Err(e) => {
				tracing::error!(
					"Failed to initialize engine: {}. Retrying in {:?}",
					e,
					ENGINE_INIT_RETRY_DELAY
				);

				thread::sleep(ENGINE_INIT_RETRY_DELAY);
			}
		}
	}

	tracing::debug!("Analysis loop shutting down");
}
