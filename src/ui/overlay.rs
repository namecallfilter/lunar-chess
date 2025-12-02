use std::{
	num::NonZeroU32,
	rc::Rc,
	time::{Duration, Instant},
};

use anyhow::Result;
use raqote::{DrawTarget, SolidSource};
use winit::{
	application::ApplicationHandler,
	dpi::{PhysicalPosition, PhysicalSize},
	event::WindowEvent,
	event_loop::{ActiveEventLoop, EventLoop},
	window::{Window, WindowId, WindowLevel},
};

use crate::{
	chess::ChessMove,
	config::CONFIG,
	model::detected::{DetectedBoard, DetectedPiece},
	ui::{
		Color, ScreenDimensions,
		draw_board::{draw_board_outline, draw_chess_grid, draw_move_arrow, draw_piece_labels},
	},
};

const UI_REDRAW_INTERVAL: Duration = Duration::from_millis(100);

const ARROW_MIN_OPACITY: u8 = 40;
const ARROW_MAX_OPACITY: u8 = 215;

const COLOR_ARROW_BASE: Color = Color::rgb(100, 150, 255); // Blue

#[derive(Debug, Clone)]
pub enum UserEvent {
	UpdateDetections(Option<DetectedBoard>, Vec<DetectedPiece>),
	UpdateBestMoves(Vec<ChessMove>),
}

struct OverlayResources {
	window: Rc<Window>,
	surface: softbuffer::Surface<Rc<Window>, Rc<Window>>,
	_context: softbuffer::Context<Rc<Window>>,
	draw_target: DrawTarget,
}

pub struct OverlayWindow {
	resources: Option<OverlayResources>,
	board: Option<DetectedBoard>,
	pieces: Vec<DetectedPiece>,
	best_moves: Vec<ChessMove>,
	screen_size: ScreenDimensions,
	should_redraw: bool,
	last_tick: Instant,
}

impl OverlayWindow {
	pub fn new() -> Self {
		Self {
			resources: None,
			board: None,
			pieces: Vec::new(),
			best_moves: Vec::new(),
			screen_size: ScreenDimensions::default(),
			should_redraw: false,
			last_tick: Instant::now(),
		}
	}

	pub fn update_detections(&mut self, board: Option<DetectedBoard>, pieces: Vec<DetectedPiece>) {
		self.board = board;
		self.pieces = pieces;
		self.should_redraw = true;
	}

	pub fn update_best_moves(&mut self, best_moves: Vec<ChessMove>) {
		self.best_moves = best_moves;
		self.should_redraw = true;
	}

	pub fn set_screen_size(&mut self, screen_size: ScreenDimensions) {
		self.screen_size = screen_size;
	}

	fn draw(&mut self) {
		if let (Some(resources), Some(board)) = (&mut self.resources, self.board.as_ref()) {
			let dt = &mut resources.draw_target;
			dt.clear(SolidSource::from_unpremultiplied_argb(0, 0, 0, 0));

			if CONFIG.debugging.show_grid {
				draw_board_outline(dt, &board.rect);
				draw_chess_grid(dt, &board.rect);
			}

			if CONFIG.debugging.show_piece_labels {
				draw_piece_labels(dt, &self.pieces);
			}

			if !self.best_moves.is_empty() {
				let max_score = self
					.best_moves
					.iter()
					.map(|m| m.score.to_numeric())
					.max()
					.unwrap_or(0);
				let min_score = self
					.best_moves
					.iter()
					.map(|m| m.score.to_numeric())
					.min()
					.unwrap_or(0);
				let score_range = (max_score - min_score) as f32;

				for best_move in &self.best_moves {
					let opacity = if score_range > 0.0 {
						let normalized =
							(best_move.score.to_numeric() - min_score) as f32 / score_range;
						let opacity_range = (ARROW_MAX_OPACITY - ARROW_MIN_OPACITY) as f32;
						ARROW_MIN_OPACITY + (normalized * opacity_range) as u8
					} else {
						ARROW_MAX_OPACITY
					};

					let arrow_color = COLOR_ARROW_BASE.with_alpha(opacity);
					let arrow_solid = arrow_color.to_solid_source();

					draw_move_arrow(dt, board, best_move, arrow_solid);
				}
			}
		}
	}

	fn render(&mut self) {
		if let Some(resources) = &mut self.resources {
			let data = resources.draw_target.get_data();

			let mut buffer = match resources.surface.buffer_mut() {
				Ok(b) => b,
				Err(e) => {
					tracing::warn!("Failed to get surface buffer: {}", e);
					return;
				}
			};

			if buffer.len() != data.len() {
				tracing::warn!(
					"Buffer length mismatch: buffer={}, data={}",
					buffer.len(),
					data.len()
				);
			}

			let len = buffer.len().min(data.len());
			buffer[..len].copy_from_slice(&data[..len]);

			if let Err(e) = buffer.present() {
				tracing::warn!("Failed to present buffer: {}", e);
			}
		}
	}
}

impl ApplicationHandler<UserEvent> for OverlayWindow {
	fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
		match event {
			UserEvent::UpdateDetections(board, pieces) => {
				self.update_detections(board, pieces);

				if let Some(resources) = &self.resources {
					resources.window.request_redraw();
				}
			}
			UserEvent::UpdateBestMoves(best_moves) => {
				self.update_best_moves(best_moves);

				if let Some(resources) = &self.resources {
					resources.window.request_redraw();
				}
			}
		}
	}

	fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
		if self.last_tick.elapsed() >= UI_REDRAW_INTERVAL {
			self.last_tick = Instant::now();

			if let Some(resources) = &self.resources {
				resources.window.request_redraw();
			}
		}
	}

	fn resumed(&mut self, event_loop: &ActiveEventLoop) {
		if self.resources.is_none() {
			let window_attributes = Window::default_attributes()
				.with_title("Lunar Chess Overlay")
				.with_transparent(true)
				.with_decorations(false)
				.with_visible(true)
				.with_inner_size(PhysicalSize::new(
					self.screen_size.width,
					self.screen_size.height,
				))
				.with_window_level(WindowLevel::AlwaysOnTop)
				.with_position(PhysicalPosition::new(0, 0));

			let window = match event_loop.create_window(window_attributes) {
				Ok(w) => Rc::new(w),
				Err(e) => {
					tracing::warn!("Failed to create window: {}", e);
					return;
				}
			};

			#[cfg(target_os = "windows")]
			{
				use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
				if let Err(e) = window.set_cursor_hittest(false) {
					tracing::debug!("Failed to set cursor hittest: {}", e);
				}

				unsafe {
					use windows::Win32::{
						Foundation::HWND,
						UI::WindowsAndMessaging::{
							SetWindowDisplayAffinity, WDA_EXCLUDEFROMCAPTURE,
						},
					};

					match window.window_handle() {
						Ok(handle) => {
							if let RawWindowHandle::Win32(handle) = handle.as_raw()
								&& CONFIG.debugging.stream_proof
							{
								let hwnd = HWND(handle.hwnd.get() as *mut _);
								let _ = SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE);
							}
						}
						Err(e) => tracing::debug!("Failed to get window handle: {}", e),
					}
				}
			}

			let context = match softbuffer::Context::new(window.clone()) {
				Ok(c) => c,
				Err(e) => {
					tracing::warn!("Failed to create softbuffer context: {}", e);
					return;
				}
			};

			let mut surface = match softbuffer::Surface::new(&context, window.clone()) {
				Ok(s) => s,
				Err(e) => {
					tracing::warn!("Failed to create softbuffer surface: {}", e);
					return;
				}
			};

			if let (Some(width), Some(height)) = (
				NonZeroU32::new(self.screen_size.width),
				NonZeroU32::new(self.screen_size.height),
			) {
				if let Err(e) = surface.resize(width, height) {
					tracing::warn!("Failed to resize surface: {}", e);
					return;
				}
			} else {
				tracing::warn!(
					"Invalid screen dimensions: {}x{}",
					self.screen_size.width,
					self.screen_size.height
				);
				return;
			}

			let draw_target = DrawTarget::new(
				self.screen_size.width as i32,
				self.screen_size.height as i32,
			);

			self.resources = Some(OverlayResources {
				window: window.clone(),
				surface,
				_context: context,
				draw_target,
			});

			window.request_redraw();
		}
	}

	fn window_event(
		&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent,
	) {
		match event {
			WindowEvent::CloseRequested => {
				event_loop.exit();
			}
			WindowEvent::RedrawRequested => {
				self.draw();
				self.render();
			}
			WindowEvent::Resized(size) => {
				if let Some(resources) = &mut self.resources
					&& let (Some(width), Some(height)) =
						(NonZeroU32::new(size.width), NonZeroU32::new(size.height))
				{
					match resources.surface.resize(width, height) {
						Ok(_) => {
							resources.draw_target =
								DrawTarget::new(size.width as i32, size.height as i32);
							resources.window.request_redraw();
						}
						Err(e) => {
							tracing::warn!("Failed to resize surface: {}", e);
						}
					}
				}
			}
			_ => {}
		}
	}
}

pub fn start_overlay(
	screen_size: ScreenDimensions,
) -> Result<(EventLoop<UserEvent>, OverlayWindow)> {
	let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;

	let mut app = OverlayWindow::new();
	app.set_screen_size(screen_size);

	Ok((event_loop, app))
}
