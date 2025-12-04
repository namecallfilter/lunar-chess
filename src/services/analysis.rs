use std::{
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
	},
	thread::{self, JoinHandle},
	time::{Duration, Instant},
};

use parking_lot::Condvar;

use crate::{
	chess::{ChessMove, board},
	engine::manager::{EngineCommand, EngineManager},
	errors::Fen,
	ui::{SharedBoardState, UserEvent},
};

const SHUTDOWN_CHECK_INTERVAL: Duration = Duration::from_secs(1);

pub struct AnalysisService {
	thread_handle: Option<JoinHandle<()>>,
	shutdown: Arc<AtomicBool>,
	board_changed: Arc<Condvar>,
}

impl AnalysisService {
	pub fn spawn(
		proxy: winit::event_loop::EventLoopProxy<UserEvent>, board_state: SharedBoardState,
		board_changed: Arc<Condvar>,
	) -> Self {
		let shutdown = Arc::new(AtomicBool::new(false));
		let shutdown_clone = Arc::clone(&shutdown);
		let board_changed_clone = Arc::clone(&board_changed);

		let handle = thread::Builder::new()
			.name("analysis-service".into())
			.spawn(move || {
				run_analysis_loop(proxy, board_state, board_changed, shutdown_clone);
			})
			.expect("Failed to spawn analysis thread");

		Self {
			thread_handle: Some(handle),
			shutdown,
			board_changed: board_changed_clone,
		}
	}

	fn shutdown(&self) {
		self.shutdown.store(true, Ordering::Release);
		self.board_changed.notify_all();
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
	let mut last_seen_version: Option<u64> = None;

	let proxy_clone = proxy.clone();
	let manager = match EngineManager::new(move |moves_with_scores, player_color| {
		if let Some(player_color) = player_color
			&& !moves_with_scores.is_empty()
		{
			let best_moves: Vec<ChessMove> = moves_with_scores
				.iter()
				.filter_map(|m| {
					board::parse_move(m.notation.as_str(), player_color)
						.map(|mv| ChessMove::with_score(mv.from, mv.to, m.score))
				})
				.collect();

			if !best_moves.is_empty() {
				proxy_clone
					.send_event(UserEvent::UpdateBestMoves(best_moves))
					.ok();
			}
		}
	}) {
		Ok(m) => m,
		Err(e) => {
			tracing::error!("Failed to initialize engine manager: {}", e);
			return;
		}
	};

	tracing::info!("Analysis engine ready");

	while !shutdown.load(Ordering::Acquire) {
		let current_position = {
			let mut lock = board_state.lock();

			let current_version = lock.as_ref().map(|s| s.version);

			if current_version == last_seen_version {
				let wait_start = Instant::now();
				let _ = board_changed.wait_for(&mut lock, SHUTDOWN_CHECK_INTERVAL);
				if wait_start.elapsed() < SHUTDOWN_CHECK_INTERVAL {
					tracing::trace!("Waited {:?} for board change", wait_start.elapsed());
				}
			}

			lock.clone()
		};

		if let Some(state) = &current_position {
			last_seen_version = Some(state.version);
		}

		if shutdown.load(Ordering::Acquire) {
			break;
		}

		if let Some(state) = current_position {
			let fen = match state.to_fen() {
				Ok(f) => f,
				Err(e) => {
					tracing::warn!("Cannot generate FEN: {}", e);
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
					tracing::warn!("Board orientation unknown, skipping analysis");
					continue;
				}
			};

			manager.send(EngineCommand::StopSearch);
			manager.send(EngineCommand::SetPosition(fen.clone(), Some(player_color)));
			manager.send(EngineCommand::StartSearch);

			last_analyzed_fen = Some(fen);
		}
	}

	tracing::debug!("Analysis loop shutting down");
}
