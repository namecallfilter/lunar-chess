use std::sync::Arc;

use anyhow::Result;
use parking_lot::{Condvar, Mutex};

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
// TODO: Premove forced mate

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
	let screen_size = capture.dimensions();

	let (event_loop, mut app) = start_overlay(screen_size)?;

	tracing::info!("Initialized in {:?}", init_start.elapsed());

	let board_state = Arc::new(Mutex::new(None));
	let board_changed = Arc::new(Condvar::new());

	let analysis_proxy = event_loop.create_proxy();
	let _analysis_service =
		AnalysisService::spawn(analysis_proxy, board_state.clone(), board_changed.clone());

	let detection_proxy = event_loop.create_proxy();
	let _detection_service = DetectionService::spawn(detection_proxy, board_state, board_changed);

	event_loop.run_app(&mut app)?;

	Ok(())
}
