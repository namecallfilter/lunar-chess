use std::sync::{Arc, Mutex};

use anyhow::Result;

mod board;
mod capture;
mod detection;
mod drawing;
mod engine;
mod error;
mod services;

use capture::ScreenCapture;
use drawing::start_overlay;
use services::{AnalysisService, DetectionService};

// TODO: Debug mode

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
	tracing::debug!("Overlay setup took {:?}", overlay_start.elapsed());

	tracing::info!("Initialization complete in {:?}", init_start.elapsed());

	let board_state = Arc::new(Mutex::new(
		Option::<(drawing::DetectedBoard, Vec<drawing::DetectedPiece>)>::None,
	));
	let board_state_for_analysis = Arc::clone(&board_state);

	tracing::info!("Starting background services...");

	let stockfish_proxy = event_loop.create_proxy();
	let _analysis_service = AnalysisService::spawn(stockfish_proxy, board_state_for_analysis);

	let detection_proxy = event_loop.create_proxy();
	let _detection_service = DetectionService::spawn(detection_proxy, board_state);

	tracing::info!("Running event loop...");
	event_loop.run_app(&mut app)?;

	Ok(())
}
