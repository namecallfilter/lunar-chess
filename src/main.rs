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
	println!("Initializing Lunar Chess...");

	let capture = ScreenCapture::new()?;
	let screen_width = capture.width();
	let screen_height = capture.height();

	println!("Screen: {}x{}", screen_width, screen_height);

	let (event_loop, mut app) = start_overlay(screen_width, screen_height)?;
	let proxy = event_loop.create_proxy();

	thread::spawn(move || {
		let capture = match ScreenCapture::new() {
			Ok(c) => c,
			Err(e) => {
				eprintln!("Failed to initialize capture in detection thread: {}", e);
				return;
			}
		};

		let mut detector = match ChessDetector::new() {
			Ok(d) => d,
			Err(e) => {
				eprintln!("Failed to initialize detector: {}", e);
				return;
			}
		};

		println!("Starting detection loop (updating every 1 second)...");

		loop {
			let image = match capture.capture() {
				Ok(img) => img,
				Err(e) => {
					eprintln!("Failed to capture screen: {}", e);
					thread::sleep(Duration::from_secs(1));
					continue;
				}
			};

			let board = match detector.detect_board(&image) {
				Ok(Some(b)) => {
					println!(
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
					proxy
						.send_event(UserEvent::UpdateDetections(None, Vec::new()))
						.ok();
					thread::sleep(Duration::from_secs(1));
					continue;
				}
				Err(e) => {
					eprintln!("Board detection error: {}", e);
					thread::sleep(Duration::from_secs(1));
					continue;
				}
			};

			let pieces = match detector.detect_pieces(&image, &board) {
				Ok(p) => {
					println!("Detected {} pieces", p.len());
					p
				}
				Err(e) => {
					eprintln!("Piece detection error: {}", e);
					Vec::new()
				}
			};

			let bounds = BoardBounds {
				x: board.x,
				y: board.y,
				width: board.width,
				height: board.height,
			};

			proxy
				.send_event(UserEvent::UpdateDetections(Some(bounds), pieces))
				.ok();

			thread::sleep(Duration::from_secs(1));
		}
	});

	event_loop.run_app(&mut app)?;

	Ok(())
}
