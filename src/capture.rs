use std::{
	sync::{
		Arc,
		atomic::{AtomicBool, AtomicU8, Ordering},
	},
	time::Duration,
};

use anyhow::Result;
use image::RgbaImage;
use parking_lot::{Mutex, MutexGuard};
use scap::{
	capturer::{Capturer, Options},
	frame::{Frame, FrameType, VideoFrame},
};

use crate::errors::CaptureError;

pub struct ScreenCapture {
	inner: Arc<InnerCapture>,
}

struct DoubleBuffer {
	buffers: [Mutex<Arc<RgbaImage>>; 2],
	active: AtomicU8,
}

impl DoubleBuffer {
	fn new(width: u32, height: u32) -> Self {
		let img = RgbaImage::new(width, height);
		Self {
			buffers: [Mutex::new(Arc::new(img.clone())), Mutex::new(Arc::new(img))],
			active: AtomicU8::new(0),
		}
	}

	fn write_buffer(&self) -> MutexGuard<'_, Arc<RgbaImage>> {
		let write_idx = (self.active.load(Ordering::Acquire) ^ 1) as usize;
		self.buffers[write_idx].lock()
	}

	fn swap(&self) {
		self.active.fetch_xor(1, Ordering::Release);
	}

	fn read_image(&self) -> Arc<RgbaImage> {
		let read_idx = self.active.load(Ordering::Acquire) as usize;
		self.buffers[read_idx].lock().clone()
	}
}

pub struct InnerCapture {
	double_buffer: DoubleBuffer,
	ready: AtomicBool,
}

impl ScreenCapture {
	pub fn new() -> Result<Self> {
		tracing::debug!("Initializing screen capture with scap...");

		let inner = Arc::new(InnerCapture {
			double_buffer: DoubleBuffer::new(0, 0),
			ready: AtomicBool::new(false),
		});

		let inner_clone = inner.clone();
		std::thread::spawn(move || {
			if let Err(e) = run_capture_loop(inner_clone) {
				tracing::error!("Screen capture thread crashed: {}", e);
			}
		});

		Ok(Self { inner })
	}

	pub fn capture(&self) -> Result<Arc<RgbaImage>> {
		if !self.inner.ready.load(Ordering::Acquire) {
			return Err(
				CaptureError::CaptureFailed("Waiting for first frame...".to_string()).into(),
			);
		}

		Ok(self.inner.double_buffer.read_image())
	}
}

fn run_capture_loop(inner: Arc<InnerCapture>) -> Result<()> {
	let options = Options {
		fps: 60,
		target: None,
		show_cursor: false,
		show_highlight: false,
		output_type: FrameType::BGRAFrame,
		..Default::default()
	};

	let mut capturer = Capturer::build(options)?;
	capturer.start_capture();

	tracing::debug!("Capture loop started.");

	loop {
		match capturer.get_next_frame() {
			Ok(Frame::Video(VideoFrame::BGRA(frame))) => {
				let mut write_guard = inner.double_buffer.write_buffer();
				let img = Arc::make_mut(&mut *write_guard);
				let raw_data = &frame.data;

				let w = frame.width as u32;
				let h = frame.height as u32;

				if img.width() != w || img.height() != h {
					*img = RgbaImage::new(w, h);
				}

				let buf_len = (w * h * 4) as usize;
				if raw_data.len() != buf_len {
					continue;
				}

				for (src, dst) in raw_data.chunks_exact(4).zip(img.chunks_exact_mut(4)) {
					dst[0] = src[2]; // R (from B position)
					dst[1] = src[1]; // G
					dst[2] = src[0]; // B (from R position)
					dst[3] = src[3]; // A
				}

				drop(write_guard);
				inner.double_buffer.swap();
				inner.ready.store(true, Ordering::Release);
			}
			Ok(_) => {}
			Err(e) => {
				tracing::warn!("Capture error: {:?}", e);
				std::thread::sleep(Duration::from_millis(100));
			}
		}
	}
}
