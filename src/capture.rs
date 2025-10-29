use anyhow::{Context, Result};
use image::RgbaImage;
use xcap::Monitor;

pub struct ScreenCapture {
	monitor: Monitor,
}

impl ScreenCapture {
	pub fn new() -> Result<Self> {
		let monitor = get_primary_monitor()?;
		tracing::debug!(
			"Screen capture initialized for monitor: {}x{}",
			monitor.width().unwrap_or(0),
			monitor.height().unwrap_or(0)
		);
		Ok(Self { monitor })
	}

	pub fn capture(&self) -> Result<RgbaImage> {
		self.monitor
			.capture_image()
			.context("Failed to capture screen image")
	}

	pub fn width(&self) -> u32 {
		self.monitor.width().unwrap_or(1920)
	}

	pub fn height(&self) -> u32 {
		self.monitor.height().unwrap_or(1080)
	}
}

fn get_primary_monitor() -> Result<Monitor> {
	let monitors = Monitor::all().context("Failed to enumerate monitors")?;

	monitors
		.into_iter()
		.find(|monitor| monitor.is_primary().unwrap_or(false))
		.context("No primary monitor found")
}
