use std::{thread, time::Duration};

use anyhow::Result;

mod capture;
mod detection;
mod drawing;

use capture::ScreenCapture;
use detection::ChessDetector;
use drawing::{BoardBounds, UserEvent, start_overlay};

#[tokio::main]
async fn main() -> Result<()> {
	tracing_subscriber::fmt()
		.with_env_filter(
			tracing_subscriber::EnvFilter::from_default_env()
				.add_directive(tracing::Level::DEBUG.into()),
		)
		.with_target(false)
		.with_thread_ids(true)
		.init();

	tracing::info!("Lunar Chess Starting");

	let init_start = std::time::Instant::now();

	tracing::info!("Initializing screen capture...");
	let capture = ScreenCapture::new()?;
	let screen_width = capture.width();
	let screen_height = capture.height();
	tracing::info!("Screen: {}x{}", screen_width, screen_height);

	tracing::info!("Starting overlay window...");
	let overlay_start = std::time::Instant::now();
	let (event_loop, mut app) = start_overlay(screen_width, screen_height)?;
	let proxy = event_loop.create_proxy();
	tracing::info!("Overlay setup took {:?}", overlay_start.elapsed());

	tracing::info!("Total initialization took {:?}", init_start.elapsed());

	thread::spawn(move || {
		tracing::info!("Detection thread started");

		let capture = match ScreenCapture::new() {
			Ok(c) => c,
			Err(e) => {
				tracing::error!("Failed to initialize capture in detection thread: {}", e);
				return;
			}
		};

		let mut detector = match ChessDetector::new() {
			Ok(d) => d,
			Err(e) => {
				tracing::error!("Failed to initialize detector: {}", e);
				return;
			}
		};

		tracing::info!("Starting detection loop (updating every 1 second)...");
		let mut iteration = 0;

		loop {
			let loop_start = std::time::Instant::now();
			iteration += 1;
			tracing::info!("Iteration {}", iteration);

			let capture_start = std::time::Instant::now();
			let image = match capture.capture() {
				Ok(img) => img,
				Err(e) => {
					tracing::error!("Failed to capture screen: {}", e);
					thread::sleep(Duration::from_secs(1));
					continue;
				}
			};
			tracing::debug!("Screen capture took {:?}", capture_start.elapsed());

			let board = match detector.detect_board(&image) {
				Ok(Some(b)) => {
					tracing::info!(
						"Board detected: ({:.0}, {:.0}) {}x{} - conf: {:.1}%",
						b.x,
						b.y,
						b.width,
						b.height,
						b.confidence * 100.0
					);
					b
				}
				Ok(None) => {
					tracing::warn!("No board detected");
					proxy
						.send_event(UserEvent::UpdateDetections(None, Vec::new()))
						.ok();
					thread::sleep(Duration::from_secs(1));
					continue;
				}
				Err(e) => {
					tracing::error!("Board detection error: {}", e);
					thread::sleep(Duration::from_secs(1));
					continue;
				}
			};

			let pieces = match detector.detect_pieces(&image, &board) {
				Ok(p) => {
					tracing::info!("Detected {} pieces", p.len());
					p
				}
				Err(e) => {
					tracing::error!("Piece detection error: {}", e);
					Vec::new()
				}
			};

			let update_start = std::time::Instant::now();
			let bounds = BoardBounds {
				x: board.x,
				y: board.y,
				width: board.width,
				height: board.height,
			};

			proxy
				.send_event(UserEvent::UpdateDetections(Some(bounds), pieces))
				.ok();
			tracing::debug!("UI update event sent in {:?}", update_start.elapsed());

			let total_time = loop_start.elapsed();
			tracing::info!("Total iteration time: {:?}\n", total_time);

			thread::sleep(Duration::from_secs(1));
		}
	});

	tracing::info!("Running event loop...");
	event_loop.run_app(&mut app)?;

	Ok(())
}
