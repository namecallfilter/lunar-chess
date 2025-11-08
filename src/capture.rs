use anyhow::Result;
use image::RgbaImage;
use xcap::Monitor;

use crate::error::CaptureError;

const DEFAULT_SCREEN_WIDTH: u32 = 1080;
const DEFAULT_SCREEN_HEIGHT: u32 = 1920;

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
			.map_err(|e| CaptureError::CaptureFailed(e.to_string()).into())
	}

	pub fn width(&self) -> u32 {
		self.monitor.width().unwrap_or(DEFAULT_SCREEN_WIDTH)
	}

	pub fn height(&self) -> u32 {
		self.monitor.height().unwrap_or(DEFAULT_SCREEN_HEIGHT)
	}
}

fn get_primary_monitor() -> Result<Monitor> {
	let monitors = Monitor::all().map_err(|e| {
		CaptureError::InvalidMonitor(format!("Failed to enumerate monitors: {}", e))
	})?;

	if monitors.is_empty() {
		return Err(CaptureError::NoMonitorsFound.into());
	}

	monitors
		.into_iter()
		.find(|monitor| monitor.is_primary().unwrap_or(false))
		.ok_or_else(|| CaptureError::NoPrimaryMonitor.into())
}
