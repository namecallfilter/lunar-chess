use std::{
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
	},
	thread::{self, JoinHandle},
	time::{Duration, Instant},
};

use anyhow::Result;
use parking_lot::{Condvar, Mutex};
use winit::event_loop::EventLoopProxy;

use crate::{
	capture::screen::ScreenCapture,
	model::detected::{BoardOrientation, BoardState, DetectedBoard},
	ui::{SharedBoardState, UserEvent},
	vision::detector::ChessDetector,
};

const DETECTION_INTERVAL: Duration = Duration::from_millis(100);
const BOARD_DETECTION_INTERVAL_FRAMES: usize = 30;

pub struct DetectionService {
	thread_handle: Option<JoinHandle<()>>,
	shutdown: Arc<AtomicBool>,
	wake: Arc<Condvar>,
	wake_lock: Arc<Mutex<bool>>,
}

impl DetectionService {
	pub fn spawn(
		proxy: EventLoopProxy<UserEvent>, board_state: SharedBoardState,
		board_changed: Arc<Condvar>,
	) -> Self {
		let shutdown = Arc::new(AtomicBool::new(false));
		let shutdown_clone = Arc::clone(&shutdown);

		let wake = Arc::new(Condvar::new());
		let wake_lock = Arc::new(Mutex::new(false));

		let wake_clone = Arc::clone(&wake);
		let wake_lock_clone = Arc::clone(&wake_lock);

		let handle = thread::Builder::new()
			.name("detection-service".into())
			.spawn(move || {
				if let Err(e) = run_detection_loop(
					proxy,
					board_state,
					board_changed,
					shutdown_clone,
					wake_clone,
					wake_lock_clone,
				) {
					tracing::error!("Detection service error: {}", e);
				}
			})
			.expect("Failed to spawn detection thread");

		Self {
			thread_handle: Some(handle),
			shutdown,
			wake,
			wake_lock,
		}
	}

	fn shutdown(&self) {
		self.shutdown.store(true, Ordering::Release);
		let mut lock = self.wake_lock.lock();
		*lock = true;
		self.wake.notify_all();
	}
}

impl Drop for DetectionService {
	fn drop(&mut self) {
		self.shutdown();
		if let Some(handle) = self.thread_handle.take() {
			let _ = handle.join();
		}
	}
}

fn run_detection_loop(
	proxy: winit::event_loop::EventLoopProxy<UserEvent>, board_state: SharedBoardState,
	board_changed: Arc<Condvar>, shutdown: Arc<AtomicBool>, wake: Arc<Condvar>,
	wake_lock: Arc<Mutex<bool>>,
) -> Result<()> {
	let capture = ScreenCapture::new()?;

	let mut detector = ChessDetector::new()?;

	tracing::debug!(
		"Detection loop started (interval: {:?})",
		DETECTION_INTERVAL
	);
	let mut detected_orientation: Option<BoardOrientation> = None;

	let mut cached_board: Option<DetectedBoard> = None;
	let mut frames_since_board_detection = BOARD_DETECTION_INTERVAL_FRAMES;
	let mut was_at_start_pos = false;

	while !shutdown.load(Ordering::Acquire) {
		let loop_start = Instant::now();

		let image = match capture.capture() {
			Ok(img) => img,
			Err(e) => {
				tracing::error!("Screen capture failed: {}", e);
				{
					let mut lock = wake_lock.lock();
					wake.wait_for(&mut lock, DETECTION_INTERVAL);
				}
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
						b.x() as i32,
						b.y() as i32,
						b.width() as i32,
						b.height() as i32
					);

					cached_board = Some(b);
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
			cached_board
		};

		let board = match board {
			Some(b) => b,
			None => {
				proxy
					.send_event(UserEvent::UpdateDetections(None, Vec::new()))
					.ok();

				{
					let mut lock = wake_lock.lock();
					wake.wait_for(&mut lock, DETECTION_INTERVAL);
				}
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
		if !pieces.is_empty() && detected_orientation.is_none() {
			let orientation = BoardState::calculate_orientation(&board, &pieces);

			detected_orientation = Some(orientation);

			if let Some(color) = orientation.to_player_color() {
				tracing::info!("Playing as {}", color);
			}
		}

		board.orientation = detected_orientation.unwrap_or(BoardOrientation::Unknown);

		if !pieces.is_empty()
			&& board.orientation != BoardOrientation::Unknown
			&& let Ok(fen) = BoardState::calculate_fen(&board, &pieces)
		{
			let fen_str = fen.as_str();
			let is_start = fen_str.starts_with("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
				|| fen_str.starts_with("RNBKQBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbkqbnr");

			if is_start {
				if !was_at_start_pos {
					tracing::info!("Starting position detected, resetting orientation");
					detected_orientation = None;
					was_at_start_pos = true;
				}
			} else {
				was_at_start_pos = false;
			}
		}

		if !pieces.is_empty() && pieces.len() <= crate::vision::detector::MAX_PIECES {
			let mut state_lock = board_state.lock();
			*state_lock = Some(BoardState::new(board, pieces.clone()));
			board_changed.notify_all();
		}

		if let Err(e) = proxy.send_event(UserEvent::UpdateDetections(Some(board), pieces)) {
			tracing::warn!("Failed to send UI update: {}", e);
		}

		let total_time = loop_start.elapsed();

		if total_time < DETECTION_INTERVAL {
			let mut lock = wake_lock.lock();
			wake.wait_for(&mut lock, DETECTION_INTERVAL - total_time);
		}
	}

	tracing::debug!("Detection loop shutting down");
	Ok(())
}
