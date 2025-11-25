use std::sync::{Arc, Mutex};

use anyhow::Result;

mod capture;
mod chess;
mod config;
mod engine;
mod errors;
mod model;
mod services;
mod ui;
mod vision;

use capture::screen::ScreenCapture;
use config::CONFIG;
use services::{AnalysisService, DetectionService};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};
use ui::start_overlay;

// TODO: Auto play
// TODO: Detect whose turn it is (autoplay only feature?)

fn main() -> Result<()> {
	tracing_subscriber::registry()
		.with(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
			EnvFilter::new(format!(
				"{}={}",
				env!("CARGO_CRATE_NAME"),
				CONFIG.debugging.level
			))
		}))
		.with(
			fmt::layer()
				.with_target(true)
				.with_thread_ids(false)
				.with_file(true)
				.with_line_number(true)
				.with_writer(std::io::stderr)
				.without_time(),
		)
		.init();

	tracing::info!("Lunar Chess starting");

	let init_start = std::time::Instant::now();

	let capture = ScreenCapture::new()?;
	let screen_width = capture.width();
	let screen_height = capture.height();

	let (event_loop, mut app) = start_overlay(screen_width, screen_height)?;

	tracing::info!("Initialized in {:?}", init_start.elapsed());

	let board_state = Arc::new(Mutex::new(
		Option::<(
			model::detected::DetectedBoard,
			Vec<model::detected::DetectedPiece>,
		)>::None,
	));
	let board_state_for_analysis = Arc::clone(&board_state);

	let analysis_proxy = event_loop.create_proxy();
	let _analysis_service = AnalysisService::spawn(analysis_proxy, board_state_for_analysis);

	let detection_proxy = event_loop.create_proxy();
	let _detection_service = DetectionService::spawn(detection_proxy, board_state);

	event_loop.run_app(&mut app)?;

	Ok(())
}
