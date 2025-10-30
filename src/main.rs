use std::{
	sync::{Arc, Mutex},
	thread,
	time::Duration,
};

use anyhow::Result;

mod board;
mod capture;
mod detection;
mod drawing;
mod stockfish;

use capture::ScreenCapture;
use detection::ChessDetector;
use drawing::{BoardBounds, UserEvent, start_overlay};

fn main() -> Result<()> {
	tracing_subscriber::fmt()
		.with_env_filter(
			tracing_subscriber::EnvFilter::from_default_env()
				.add_directive("lunar_chess=debug".parse().unwrap())
				.add_directive(tracing::Level::WARN.into()),
		)
		.with_target(false)
		.with_thread_ids(true)
		.init();

	tracing::info!("Lunar Chess starting");

	let init_start = std::time::Instant::now();

	tracing::info!("Initializing screen capture...");
	let capture = ScreenCapture::new()?;
	let screen_width = capture.width();
	let screen_height = capture.height();
	tracing::info!("Screen resolution: {}x{}", screen_width, screen_height);

	tracing::info!("Starting overlay window...");
	let overlay_start = std::time::Instant::now();
	let (event_loop, mut app) = start_overlay(screen_width, screen_height)?;
	let proxy = event_loop.create_proxy();
	tracing::debug!("Overlay setup took {:?}", overlay_start.elapsed());

	tracing::info!("Initialization complete in {:?}", init_start.elapsed());

	let current_fen = Arc::new(Mutex::new(
		Option::<(drawing::DetectedBoard, Vec<drawing::DetectedPiece>)>::None,
	));
	let fen_for_stockfish = Arc::clone(&current_fen);

	let stockfish_proxy = event_loop.create_proxy();

	tracing::info!("Initializing Stockfish...");
	std::thread::spawn(move || {
		let mut last_analyzed_fen: Option<String> = None;

		loop {
			let stockfish_result = stockfish::StockfishWrapper::new();

			match stockfish_result {
				Ok(mut stockfish) => {
					tracing::info!("Stockfish analysis thread ready");

					loop {
						thread::sleep(Duration::from_secs(2));

						let board_state = fen_for_stockfish.lock().unwrap().clone();

						if let Some((board, pieces)) = board_state {
							let fen = board::to_fen(&board, &pieces);

							if last_analyzed_fen.as_ref() == Some(&fen) {
								tracing::trace!("Position unchanged, skipping analysis");
								continue;
							}

							tracing::debug!("Valid FEN: {}", fen);

							match stockfish.set_position(&fen) {
								Ok(_) => {
									tracing::trace!("Position set successfully");
								}
								Err(e) => {
									tracing::error!(
										"Stockfish error on position '{}': {:?}",
										fen,
										e
									);
									tracing::warn!("Stockfish may have crashed, restarting...");
									break;
								}
							}

							match stockfish.get_best_moves() {
								Ok(moves_with_scores) => {
									if !moves_with_scores.is_empty() {
										tracing::info!(
											"Top {} moves: {}",
											moves_with_scores.len(),
											moves_with_scores
												.iter()
												.map(|m| format!("{} ({})", m.move_str, m.score))
												.collect::<Vec<_>>()
												.join(", ")
										);

										let best_moves: Vec<drawing::BestMove> = moves_with_scores
											.iter()
											.filter_map(|m| {
												board::parse_move(&m.move_str).map(|mut mv| {
													mv.score = m.score;
													mv
												})
											})
											.collect();

										if !best_moves.is_empty() {
											stockfish_proxy
												.send_event(UserEvent::UpdateBestMoves(best_moves))
												.ok();
										}
									}

									last_analyzed_fen = Some(fen);
								}
								Err(e) => {
									tracing::error!("Failed to get best moves: {:?}", e);
									tracing::warn!("Stockfish may have crashed, restarting...");
									break;
								}
							}
						} else {
							tracing::trace!("No board state available for analysis");
						}
					}

					tracing::warn!("Stockfish crashed, attempting restarting...");
					thread::sleep(Duration::from_secs_f32(0.5));
				}
				Err(e) => {
					tracing::error!("Failed to initialize Stockfish: {}", e);
					tracing::warn!("Retrying Stockfish initialization in 5 seconds...");
					thread::sleep(Duration::from_secs(5));
				}
			}
		}
	});

	thread::spawn(move || {
		tracing::debug!("Detection thread started");

		let capture = match ScreenCapture::new() {
			Ok(c) => c,
			Err(e) => {
				tracing::error!(
					"Failed to initialize screen capture in detection thread: {}",
					e
				);
				return;
			}
		};

		let mut detector = match ChessDetector::new() {
			Ok(d) => d,
			Err(e) => {
				tracing::error!("Failed to initialize chess detector: {}", e);
				return;
			}
		};

		tracing::info!("Detection loop started (polling interval: 1s)");
		let mut iteration = 0;

		loop {
			let loop_start = std::time::Instant::now();
			iteration += 1;
			tracing::debug!("Detection iteration {}", iteration);

			let capture_start = std::time::Instant::now();
			let image = match capture.capture() {
				Ok(img) => img,
				Err(e) => {
					tracing::error!("Screen capture failed: {}", e);
					thread::sleep(Duration::from_secs(1));
					continue;
				}
			};
			tracing::trace!("Screen capture took {:?}", capture_start.elapsed());

			let board = match detector.detect_board(&image) {
				Ok(Some(b)) => {
					tracing::debug!(
						"Board detected at ({:.0}, {:.0}) size {}x{} (confidence: {:.1}%)",
						b.x,
						b.y,
						b.width,
						b.height,
						b.confidence * 100.0
					);
					b
				}
				Ok(None) => {
					tracing::debug!("No board detected in current frame");
					proxy
						.send_event(UserEvent::UpdateDetections(None, Vec::new()))
						.ok();
					thread::sleep(Duration::from_secs(1));
					continue;
				}
				Err(e) => {
					tracing::error!("Board detection failed: {}", e);
					thread::sleep(Duration::from_secs(1));
					continue;
				}
			};

			let pieces = match detector.detect_pieces(&image, &board) {
				Ok(p) => {
					tracing::debug!("Detected {} pieces on board", p.len());

					if p.len() > 32 {
						tracing::warn!(
							"Detected {} pieces (max 32), skipping invalid detection",
							p.len()
						);
						Vec::new()
					} else {
						p
					}
				}
				Err(e) => {
					tracing::error!("Piece detection failed: {}", e);
					Vec::new()
				}
			};

			if !pieces.is_empty() && pieces.len() <= 32 {
				let mut fen_lock = current_fen.lock().unwrap();
				*fen_lock = Some((board.clone(), pieces.clone()));
			}

			let update_start = std::time::Instant::now();
			let bounds = BoardBounds {
				x: board.x,
				y: board.y,
				width: board.width,
				height: board.height,
			};

			if let Err(e) = proxy.send_event(UserEvent::UpdateDetections(Some(bounds), pieces)) {
				tracing::error!("Failed to send UI update event: {}", e);
			}
			tracing::trace!("UI update event sent in {:?}", update_start.elapsed());

			let total_time = loop_start.elapsed();
			tracing::debug!("Iteration completed in {:?}", total_time);

			thread::sleep(Duration::from_secs(1));
		}
	});

	tracing::info!("Running event loop...");
	event_loop.run_app(&mut app)?;

	Ok(())
}
