use std::sync::Arc;

use anyhow::Result;
use image::RgbaImage;
use parking_lot::Mutex;

use crate::errors::CaptureError;

pub struct ScreenCapture {
	inner: Arc<InnerCapture>,
}

pub struct InnerCapture {
	latest_frame: Mutex<Option<RgbaImage>>,
	width: u32,
	height: u32,
}

// TODO: MacOS and Linux

impl ScreenCapture {
	pub fn new() -> Result<Self> {
		tracing::debug!("Initializing screen capture...");

		let (inner, width, height) = windows::init_capture()?;

		tracing::debug!("Capture ready ({}x{})", width, height);

		Ok(Self { inner })
	}

	pub fn capture(&self) -> Result<RgbaImage> {
		let guard = self.inner.latest_frame.lock();

		match &*guard {
			Some(img) => Ok(img.clone()),
			None => {
				Err(CaptureError::CaptureFailed("Waiting for first frame...".to_string()).into())
			}
		}
	}

	pub fn width(&self) -> u32 {
		self.inner.width
	}

	pub fn height(&self) -> u32 {
		self.inner.height
	}
}

#[cfg(target_os = "windows")]
mod windows {
	use windows_capture::{
		capture::{Context as WinContext, GraphicsCaptureApiHandler},
		frame::Frame,
		graphics_capture_api::InternalCaptureControl,
		monitor::Monitor,
		settings::{
			ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
			MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
		},
	};

	use super::*;

	struct CaptureHandler {
		frame_buffer: Arc<InnerCapture>,
	}

	impl GraphicsCaptureApiHandler for CaptureHandler {
		type Error = Box<dyn std::error::Error + Send + Sync>;
		type Flags = Arc<InnerCapture>;

		fn new(ctx: WinContext<Self::Flags>) -> Result<Self, Self::Error> {
			Ok(Self {
				frame_buffer: ctx.flags,
			})
		}

		fn on_frame_arrived(
			&mut self, frame: &mut Frame, _: InternalCaptureControl,
		) -> Result<(), Self::Error> {
			let width = frame.width();
			let height = frame.height();
			let mut buffer = frame.buffer()?;

			let raw_data = buffer.as_raw_buffer();

			let mut rgba_buffer = Vec::with_capacity(raw_data.len());

			for chunk in raw_data.chunks_exact(4) {
				rgba_buffer.push(chunk[2]);
				rgba_buffer.push(chunk[1]);
				rgba_buffer.push(chunk[0]);
				rgba_buffer.push(chunk[3]);
			}

			if let Some(img_buf) = RgbaImage::from_raw(width, height, rgba_buffer) {
				let mut guard = self.frame_buffer.latest_frame.lock();
				*guard = Some(img_buf);
			}

			Ok(())
		}

		fn on_closed(&mut self) -> Result<(), Self::Error> {
			Ok(())
		}
	}

	pub fn init_capture() -> Result<(Arc<InnerCapture>, u32, u32)> {
		let monitor = Monitor::primary()?;
		let width = monitor.width()?;
		let height = monitor.height()?;

		let inner = Arc::new(InnerCapture {
			latest_frame: Mutex::new(None),
			width,
			height,
		});

		let settings = Settings::new(
			monitor,
			CursorCaptureSettings::Default,
			DrawBorderSettings::WithoutBorder,
			SecondaryWindowSettings::Default,
			MinimumUpdateIntervalSettings::Default,
			DirtyRegionSettings::Default,
			ColorFormat::Bgra8,
			inner.clone(),
		);

		std::thread::spawn(move || {
			CaptureHandler::start(settings).expect("Screen capture thread crashed");
		});

		Ok((inner, width, height))
	}
}
