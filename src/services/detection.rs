use std::{
	thread::{self, JoinHandle},
	time::{Duration, Instant},
};

use anyhow::Result;
use winit::event_loop::EventLoopProxy;

use crate::{
	capture::screen::ScreenCapture,
	chess::board,
	model::detected::DetectedBoard,
	ui::{BoardBounds, SharedBoardState, UserEvent},
	vision::detector::ChessDetector,
};

const DETECTION_INTERVAL_SECS: f32 = 0.1;
const BOARD_DETECTION_INTERVAL_FRAMES: usize = 30;

pub struct DetectionService {
	thread_handle: Option<JoinHandle<()>>,
}

impl DetectionService {
	pub fn spawn(proxy: EventLoopProxy<UserEvent>, board_state: SharedBoardState) -> Self {
		let handle = thread::spawn(move || {
			if let Err(e) = run_detection_loop(proxy, board_state) {
				tracing::error!("Detection service error: {}", e);
			}
		});

		Self {
			thread_handle: Some(handle),
		}
	}
}

impl Drop for DetectionService {
	fn drop(&mut self) {
		if let Some(handle) = self.thread_handle.take() {
			let _ = handle.join();
		}
	}
}

fn run_detection_loop(
	proxy: winit::event_loop::EventLoopProxy<UserEvent>, board_state: SharedBoardState,
) -> Result<()> {
	tracing::debug!("Detection thread started");

	let capture = ScreenCapture::new()?;

	let mut detector = ChessDetector::new()?;

	tracing::info!(
		"Detection loop started (polling interval: {}s)",
		DETECTION_INTERVAL_SECS
	);
	let mut iteration = 0;
	let mut orientation_detected = false;
	let mut detected_playing_as_white = true;

	let mut cached_board: Option<DetectedBoard> = None;
	let mut frames_since_board_detection = BOARD_DETECTION_INTERVAL_FRAMES;

	loop {
		let loop_start = Instant::now();
		iteration += 1;
		tracing::debug!("Detection iteration {}", iteration);

		let capture_start = Instant::now();
		let image = match capture.capture() {
			Ok(img) => img,
			Err(e) => {
				tracing::error!("Screen capture failed: {}", e);
				thread::sleep(Duration::from_secs_f32(DETECTION_INTERVAL_SECS));
				continue;
			}
		};
		tracing::trace!("Screen capture took {:?}", capture_start.elapsed());

		let board = if cached_board.is_none()
			|| frames_since_board_detection >= BOARD_DETECTION_INTERVAL_FRAMES
		{
			tracing::debug!("Running full board detection (frame {})", iteration);
			match detector.detect_board(&image) {
				Ok(Some(b)) => {
					tracing::debug!(
						"Board detected at ({:.0}, {:.0}) size {}x{})",
						b.x,
						b.y,
						b.width,
						b.height,
					);
					cached_board = Some(b.clone());
					frames_since_board_detection = 0;
					Some(b)
				}
				Ok(None) => {
					tracing::debug!("No board detected in current frame");
					cached_board = None;
					None
				}
				Err(e) => {
					tracing::error!("Board detection failed: {}", e);
					cached_board = None;
					None
				}
			}
		} else {
			tracing::trace!("Using cached board detection");
			frames_since_board_detection += 1;
			cached_board.clone()
		};

		let board = match board {
			Some(b) => b,
			None => {
				proxy
					.send_event(UserEvent::UpdateDetections(None, Vec::new()))
					.ok();
				thread::sleep(Duration::from_secs_f32(DETECTION_INTERVAL_SECS));
				continue;
			}
		};

		let pieces = match detector.detect_pieces(&image, &board) {
			Ok(p) => {
				tracing::debug!("Detected {} pieces on board", p.len());

				if p.len() > crate::vision::detector::MAX_PIECES {
					tracing::warn!(
						"Detected {} pieces (max {}), skipping invalid detection",
						p.len(),
						crate::vision::detector::MAX_PIECES
					);
					Vec::new()
				} else {
					p
				}
			}
			Err(e) => {
				tracing::error!("Piece detection failed: {}", e);
				Vec::new()
			}
		};

		let mut board = board;
		if !pieces.is_empty() && !orientation_detected {
			let playing_as_white = board::detect_board_orientation(&board, &pieces);
			detected_playing_as_white = playing_as_white;
			orientation_detected = true;
			tracing::info!(
				"Board orientation detected: playing as {}",
				if playing_as_white { "white" } else { "black" }
			);
		}

		board.playing_as_white = detected_playing_as_white;

		if !pieces.is_empty() && pieces.len() <= crate::vision::detector::MAX_PIECES {
			let mut state_lock = board_state.lock().unwrap();
			*state_lock = Some((board.clone(), pieces.clone()));
		}

		let update_start = std::time::Instant::now();
		let bounds = BoardBounds {
			x: board.x,
			y: board.y,
			width: board.width,
			height: board.height,
			playing_as_white: board.playing_as_white,
		};

		if let Err(e) = proxy.send_event(UserEvent::UpdateDetections(Some(bounds), pieces)) {
			tracing::error!("Failed to send UI update event: {}", e);
		}
		tracing::trace!("UI update event sent in {:?}", update_start.elapsed());

		let total_time = loop_start.elapsed();
		tracing::debug!("Iteration completed in {:?}", total_time);

		if total_time.as_secs_f32() < DETECTION_INTERVAL_SECS {
			thread::sleep(Duration::from_secs_f32(
				DETECTION_INTERVAL_SECS - total_time.as_secs_f32(),
			));
		}
	}
}
