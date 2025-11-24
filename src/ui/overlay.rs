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
	config::CONFIG,
    model::detected::{DetectedBoard, DetectedPiece},
	ui::draw_board::{
		draw_board_outline, draw_chess_grid, draw_move_arrow,
		draw_piece_labels,
	},
};

const DEFAULT_SCREEN_WIDTH: u32 = 1920;
const DEFAULT_SCREEN_HEIGHT: u32 = 1080;

const UI_REDRAW_INTERVAL_MS: u64 = 100;

const ARROW_BASE_OPACITY: u8 = 200;
const ARROW_OPACITY_STEP: u8 = 60;
const COLOR_ARROW_BASE: (u8, u8, u8) = (100, 150, 255); // Blue

#[derive(Debug, Clone)]
pub enum UserEvent {
	UpdateDetections(Option<BoardBounds>, Vec<DetectedPiece>),
	UpdateBestMoves(Vec<BestMove>),
	#[allow(dead_code)]
	Tick,
}

#[derive(Debug, Clone, Copy)]
pub struct BestMove {
	pub from_file: u8,
	pub from_rank: u8,
	pub to_file: u8,
	pub to_rank: u8,
	pub score: i32,
}

pub struct OverlayWindow {
	window: Option<Rc<Window>>,
	surface: Option<softbuffer::Surface<Rc<Window>, Rc<Window>>>,
	context: Option<softbuffer::Context<Rc<Window>>>,
	draw_target: Option<DrawTarget>,
	board_bounds: Option<BoardBounds>,
	pieces: Vec<DetectedPiece>,
	best_moves: Vec<BestMove>,
	screen_width: u32,
	screen_height: u32,
	should_redraw: bool,
	last_tick: Instant,
}

#[derive(Clone, Copy, Debug)]
pub struct BoardBounds {
	pub x: f32,
	pub y: f32,
	pub width: f32,
	pub height: f32,
	pub playing_as_white: bool,
}

impl OverlayWindow {
	pub fn new() -> Self {
		Self {
			window: None,
			surface: None,
			context: None,
			draw_target: None,
			board_bounds: None,
			pieces: Vec::new(),
			best_moves: Vec::new(),
			screen_width: DEFAULT_SCREEN_WIDTH,
			screen_height: DEFAULT_SCREEN_HEIGHT,
			should_redraw: false,
			last_tick: Instant::now(),
		}
	}

	pub fn update_detections(&mut self, bounds: Option<BoardBounds>, pieces: Vec<DetectedPiece>) {
		self.board_bounds = bounds;
		self.pieces = pieces;
		self.should_redraw = true;
	}

	pub fn update_best_moves(&mut self, best_moves: Vec<BestMove>) {
		self.best_moves = best_moves;
		self.should_redraw = true;
	}

	pub fn set_screen_size(&mut self, width: u32, height: u32) {
		self.screen_width = width;
		self.screen_height = height;
	}

	fn draw(&mut self) {
		if let (Some(dt), Some(bounds)) = (&mut self.draw_target, self.board_bounds) {
			dt.clear(SolidSource::from_unpremultiplied_argb(0, 0, 0, 0));

			if CONFIG.debugging.show_grid {
				draw_board_outline(dt, bounds.x, bounds.y, bounds.width, bounds.height);
				draw_chess_grid(dt, bounds.x, bounds.y, bounds.width, bounds.height);
			}

			if CONFIG.debugging.show_piece_labels {
				draw_piece_labels(dt, &self.pieces);
			}

			if !self.best_moves.is_empty() {
				for (index, best_move) in self.best_moves.iter().enumerate() {
					let opacity =
						ARROW_BASE_OPACITY.saturating_sub((index as u8) * ARROW_OPACITY_STEP);

					let arrow_color = (
						opacity,
						COLOR_ARROW_BASE.0,
						COLOR_ARROW_BASE.1,
						COLOR_ARROW_BASE.2,
					);

					draw_move_arrow(
						dt,
						&DetectedBoard {
							x: bounds.x,
							y: bounds.y,
							width: bounds.width,
							height: bounds.height,
							playing_as_white: bounds.playing_as_white,
						},
						best_move.from_file,
						best_move.from_rank,
						best_move.to_file,
						best_move.to_rank,
						arrow_color,
					);
				}
			}
		}
	}

	fn render(&mut self) {
		if let (Some(surface), Some(dt)) = (&mut self.surface, &self.draw_target) {
			let data = dt.get_data();

			let mut buffer = match surface.buffer_mut() {
				Ok(b) => b,
				Err(e) => {
					tracing::error!("Failed to get surface buffer: {}", e);
					return;
				}
			};

			for (i, pixel) in data.iter().enumerate() {
				if i < buffer.len() {
					buffer[i] = *pixel;
				}
			}

			if let Err(e) = buffer.present() {
				tracing::error!("Failed to present buffer: {}", e);
			}
		}
	}
}

impl ApplicationHandler<UserEvent> for OverlayWindow {
	fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
		match event {
			UserEvent::UpdateDetections(bounds, pieces) => {
				self.update_detections(bounds, pieces);
				if let Some(window) = &self.window {
					window.request_redraw();
				}
			}
			UserEvent::UpdateBestMoves(best_moves) => {
				self.update_best_moves(best_moves);
				if let Some(window) = &self.window {
					window.request_redraw();
				}
			}
			UserEvent::Tick => {}
		}
	}

	fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
		if self.last_tick.elapsed() >= Duration::from_millis(UI_REDRAW_INTERVAL_MS) {
			self.last_tick = Instant::now();
			if let Some(window) = &self.window {
				window.request_redraw();
			}
		}
	}

	fn resumed(&mut self, event_loop: &ActiveEventLoop) {
		if self.window.is_none() {
			let window_attributes = Window::default_attributes()
				.with_title("Lunar Chess Overlay")
				.with_transparent(true)
				.with_decorations(false)
				.with_visible(true)
				.with_inner_size(PhysicalSize::new(self.screen_width, self.screen_height))
				.with_window_level(WindowLevel::AlwaysOnTop)
				.with_position(PhysicalPosition::new(0, 0));

			let window = match event_loop.create_window(window_attributes) {
				Ok(w) => Rc::new(w),
				Err(e) => {
					tracing::error!("Failed to create window: {}", e);
					return;
				}
			};

			#[cfg(target_os = "windows")]
			{
				use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
				if let Err(e) = window.set_cursor_hittest(false) {
					tracing::warn!("Failed to set cursor hittest: {}", e);
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
							if let RawWindowHandle::Win32(handle) = handle.as_raw() {
								let hwnd = HWND(handle.hwnd.get() as *mut _);
								let _ = SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE);
							}
						}
						Err(e) => tracing::warn!("Failed to get window handle: {}", e),
					}
				}
			}

			let context = match softbuffer::Context::new(window.clone()) {
				Ok(c) => c,
				Err(e) => {
					tracing::error!("Failed to create softbuffer context: {}", e);
					return;
				}
			};

			let mut surface = match softbuffer::Surface::new(&context, window.clone()) {
				Ok(s) => s,
				Err(e) => {
					tracing::error!("Failed to create softbuffer surface: {}", e);
					return;
				}
			};

			if let (Some(width), Some(height)) = (
				NonZeroU32::new(self.screen_width),
				NonZeroU32::new(self.screen_height),
			) {
				if let Err(e) = surface.resize(width, height) {
					tracing::error!("Failed to resize surface: {}", e);
					return;
				}
			} else {
				tracing::error!(
					"Invalid screen dimensions: {}x{}",
					self.screen_width,
					self.screen_height
				);
				return;
			}

			self.draw_target = Some(DrawTarget::new(
				self.screen_width as i32,
				self.screen_height as i32,
			));

			self.surface = Some(surface);
			self.context = Some(context);
			self.window = Some(window.clone());

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
				if let (Some(surface), Some(width), Some(height)) = (
					&mut self.surface,
					NonZeroU32::new(size.width),
					NonZeroU32::new(size.height),
				) {
					surface.resize(width, height).unwrap_or_else(|e| {
						tracing::error!("Failed to resize surface: {}", e);
					});
				}

				self.draw_target = Some(DrawTarget::new(size.width as i32, size.height as i32));

				if let Some(window) = &self.window {
					window.request_redraw();
				}
			}
			_ => {}
		}
	}
}

pub fn start_overlay(
	screen_width: u32, screen_height: u32,
) -> Result<(EventLoop<UserEvent>, OverlayWindow)> {
	let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;

	let mut app = OverlayWindow::new();
	app.set_screen_size(screen_width, screen_height);

	Ok((event_loop, app))
}
