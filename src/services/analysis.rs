use std::{
	thread::{self, JoinHandle},
	time::Duration,
};

use crate::{
	chess::board,
	engine::EngineWrapper,
	ui::{BestMove, SharedBoardState, UserEvent},
};

const ANALYSIS_INTERVAL_SECS: u64 = 2;
const ENGINE_RESTART_DELAY_SECS: u64 = 1;
const ENGINE_INIT_RETRY_DELAY_SECS: u64 = 5;

pub struct AnalysisService {
	thread_handle: Option<JoinHandle<()>>,
}

impl AnalysisService {
	pub fn spawn(
		proxy: winit::event_loop::EventLoopProxy<UserEvent>, board_state: SharedBoardState,
	) -> Self {
		let handle = thread::spawn(move || {
			run_analysis_loop(proxy, board_state);
		});

		Self {
			thread_handle: Some(handle),
		}
	}
}

impl Drop for AnalysisService {
	fn drop(&mut self) {
		if let Some(handle) = self.thread_handle.take() {
			let _ = handle.join();
		}
	}
}

fn run_analysis_loop(
	proxy: winit::event_loop::EventLoopProxy<UserEvent>, board_state: SharedBoardState,
) {
	let mut last_analyzed_fen: Option<String> = None;

	loop {
		let engine_result = EngineWrapper::new();

		match engine_result {
			Ok(mut engine) => {
				tracing::info!("Analysis engine ready");

				loop {
					thread::sleep(Duration::from_secs(ANALYSIS_INTERVAL_SECS));

					let current_position = board_state.lock().unwrap().clone();

					if let Some((board, pieces)) = current_position {
						let fen = board::to_fen(&board, &pieces);

						if last_analyzed_fen.as_ref() == Some(&fen) {
							continue;
						}

						match engine.set_position(&fen) {
							Ok(_) => {}
							Err(e) => {
								tracing::warn!("Engine error on position '{}': {:?}", fen, e);
								tracing::debug!("Engine may have crashed, restarting...");
								break;
							}
						}

						match engine.get_best_moves() {
							Ok(moves_with_scores) => {
								if !moves_with_scores.is_empty() {
									let best_moves: Vec<BestMove> = moves_with_scores
										.iter()
										.filter_map(|m| {
											board::parse_move(&m.move_str, board.playing_as_white)
												.map(|mut mv| {
													mv.score = m.score;
													mv
												})
										})
										.collect();

									if !best_moves.is_empty() {
										proxy
											.send_event(UserEvent::UpdateBestMoves(best_moves))
											.ok();
									}
								}

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

				tracing::warn!("Engine crashed, restarting...");
				thread::sleep(Duration::from_secs(ENGINE_RESTART_DELAY_SECS));
			}
			Err(e) => {
				tracing::error!(
					"Failed to initialize engine: {}. Retrying in {}s",
					e,
					ENGINE_INIT_RETRY_DELAY_SECS
				);
				thread::sleep(Duration::from_secs(ENGINE_INIT_RETRY_DELAY_SECS));
			}
		}
	}
}
