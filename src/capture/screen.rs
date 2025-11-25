use std::sync::{
	atomic::{AtomicUsize, Ordering},
	Arc,
};

use anyhow::Result;
use image::RgbaImage;
use parking_lot::Mutex;

use crate::errors::CaptureError;

pub struct ScreenCapture {
	inner: Arc<InnerCapture>,
}

struct DoubleBuffer {
	buffers: [Mutex<Vec<u8>>; 2],
	active: AtomicUsize,
	width: u32,
	height: u32,
}

impl DoubleBuffer {
	fn new(width: u32, height: u32) -> Self {
		let size = (width * height * 4) as usize;
		Self {
			buffers: [Mutex::new(vec![0u8; size]), Mutex::new(vec![0u8; size])],
			active: AtomicUsize::new(0),
			width,
			height,
		}
	}

	fn write_buffer(&self) -> parking_lot::MutexGuard<'_, Vec<u8>> {
		let write_idx = 1 - self.active.load(Ordering::Acquire);
		self.buffers[write_idx].lock()
	}

	fn swap(&self) {
		let current = self.active.load(Ordering::Acquire);
		self.active.store(1 - current, Ordering::Release);
	}

	fn read_image(&self) -> RgbaImage {
		let read_idx = self.active.load(Ordering::Acquire);
		let guard = self.buffers[read_idx].lock();
		RgbaImage::from_raw(self.width, self.height, guard.clone())
			.expect("Buffer size matches image dimensions")
	}
}

pub struct InnerCapture {
	double_buffer: DoubleBuffer,
	ready: std::sync::atomic::AtomicBool,
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
		if !self.inner.ready.load(Ordering::Acquire) {
			return Err(
				CaptureError::CaptureFailed("Waiting for first frame...".to_string()).into(),
			);
		}

		Ok(self.inner.double_buffer.read_image())
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
	use std::sync::atomic::Ordering;

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
			let mut buffer = frame.buffer()?;
			let raw_data = buffer.as_raw_buffer();

			{
				let mut write_buf = self.frame_buffer.double_buffer.write_buffer();

				if write_buf.len() == raw_data.len() {
					for (i, chunk) in raw_data.chunks_exact(4).enumerate() {
						let offset = i * 4;
						write_buf[offset] = chunk[2]; // R
						write_buf[offset + 1] = chunk[1]; // G
						write_buf[offset + 2] = chunk[0]; // B
						write_buf[offset + 3] = chunk[3]; // A
					}
				}
			}

			self.frame_buffer.double_buffer.swap();
			self.frame_buffer.ready.store(true, Ordering::Release);

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
			double_buffer: DoubleBuffer::new(width, height),
			ready: std::sync::atomic::AtomicBool::new(false),
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
			if let Err(e) = CaptureHandler::start(settings) {
				tracing::error!("Screen capture thread crashed: {}", e);
			}
		});

		Ok((inner, width, height))
	}
}
