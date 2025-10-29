use anyhow::Result;
use image::RgbaImage;
use xcap::{Monitor, XCapError};

pub struct ScreenCapture {
	monitor: Monitor,
}

impl ScreenCapture {
	pub fn new() -> Result<Self> {
		let monitor = get_primary_monitor()?;
		Ok(Self { monitor })
	}

	pub fn capture(&self) -> Result<RgbaImage> {
		let image = self.monitor.capture_image()?;
		Ok(image)
	}

	pub fn width(&self) -> u32 {
		self.monitor.width().unwrap_or(1920)
	}

	pub fn height(&self) -> u32 {
		self.monitor.height().unwrap_or(1080)
	}
}

fn get_primary_monitor() -> Result<Monitor> {
	let monitors = Monitor::all()?;
	monitors
		.into_iter()
		.find(|monitor| monitor.is_primary().unwrap_or(false))
		.ok_or_else(|| XCapError::Error("Could not find primary monitor".to_string()).into())
}
