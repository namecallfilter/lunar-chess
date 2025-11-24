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
	let capture = ScreenCapture::new()?;

	let mut detector = ChessDetector::new()?;

	tracing::debug!(
		"Detection loop started (interval: {}s)",
		DETECTION_INTERVAL_SECS
	);
	let mut orientation_detected = false;
	let mut detected_playing_as_white = true;

	let mut cached_board: Option<DetectedBoard> = None;
	let mut frames_since_board_detection = BOARD_DETECTION_INTERVAL_FRAMES;

	loop {
		let loop_start = Instant::now();

		let image = match capture.capture() {
			Ok(img) => img,
			Err(e) => {
				tracing::error!("Screen capture failed: {}", e);
				thread::sleep(Duration::from_secs_f32(DETECTION_INTERVAL_SECS));
				continue;
			}
		};

		let board = if cached_board.is_none()
			|| frames_since_board_detection >= BOARD_DETECTION_INTERVAL_FRAMES
		{
			match detector.detect_board(&image) {
				Ok(Some(b)) => {
					tracing::debug!(
						"Board detected at ({}, {}) size {}x{}",
						b.x as i32,
						b.y as i32,
						b.width as i32,
						b.height as i32
					);
					cached_board = Some(b.clone());
					frames_since_board_detection = 0;
					Some(b)
				}
				Ok(None) => {
					cached_board = None;
					None
				}
				Err(e) => {
					tracing::warn!("Board detection failed: {}", e);
					cached_board = None;
					None
				}
			}
		} else {
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
				if p.len() > crate::vision::detector::MAX_PIECES {
					tracing::debug!("Too many pieces detected ({}), skipping", p.len());
					Vec::new()
				} else {
					p
				}
			}
			Err(e) => {
				tracing::warn!("Piece detection failed: {}", e);
				Vec::new()
			}
		};

		let mut board = board;
		if !pieces.is_empty() && !orientation_detected {
			let playing_as_white = board::detect_board_orientation(&board, &pieces);
			detected_playing_as_white = playing_as_white;
			orientation_detected = true;
			tracing::info!(
				"Playing as {}",
				if playing_as_white { "white" } else { "black" }
			);
		}

		board.playing_as_white = detected_playing_as_white;

		if !pieces.is_empty() && pieces.len() <= crate::vision::detector::MAX_PIECES {
			let mut state_lock = board_state.lock().unwrap();
			*state_lock = Some((board.clone(), pieces.clone()));
		}

		let bounds = BoardBounds {
			x: board.x,
			y: board.y,
			width: board.width,
			height: board.height,
			playing_as_white: board.playing_as_white,
		};

		if let Err(e) = proxy.send_event(UserEvent::UpdateDetections(Some(bounds), pieces)) {
			tracing::warn!("Failed to send UI update: {}", e);
		}

		let total_time = loop_start.elapsed();

		if total_time.as_secs_f32() < DETECTION_INTERVAL_SECS {
			thread::sleep(Duration::from_secs_f32(
				DETECTION_INTERVAL_SECS - total_time.as_secs_f32(),
			));
		}
	}
}
