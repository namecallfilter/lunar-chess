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

use crate::drawing::board::{
	DetectedBoard, DetectedPiece, draw_board_outline, draw_chess_grid, draw_move_arrow,
	draw_piece_labels,
};

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
			screen_width: 1920,
			screen_height: 1080,
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

			draw_board_outline(dt, bounds.x, bounds.y, bounds.width, bounds.height);
			draw_chess_grid(dt, bounds.x, bounds.y, bounds.width, bounds.height);

			draw_piece_labels(dt, &self.pieces);

			if !self.best_moves.is_empty() {
				for (index, best_move) in self.best_moves.iter().enumerate() {
					let opacity = match index {
						0 => 255,
						1 => 180,
						2 => 110,
						_ => 80,
					};

					let arrow_color = (opacity, 100, 150, 255);

					draw_move_arrow(
						dt,
						&DetectedBoard {
							x: bounds.x,
							y: bounds.y,
							width: bounds.width,
							height: bounds.height,
							confidence: 1.0,
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

			let mut buffer = surface.buffer_mut().unwrap();

			for (i, pixel) in data.iter().enumerate() {
				if i < buffer.len() {
					buffer[i] = *pixel;
				}
			}

			buffer.present().unwrap();
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
		if self.last_tick.elapsed() >= Duration::from_millis(100) {
			self.last_tick = Instant::now();
			if let Some(window) = &self.window {
				window.request_redraw();
			}
		}
	}

	fn resumed(&mut self, event_loop: &ActiveEventLoop) {
		if self.window.is_none() {
			let window = Rc::new(
				event_loop
					.create_window(
						Window::default_attributes()
							.with_title("Lunar Chess Overlay")
							.with_transparent(true)
							.with_decorations(false)
							.with_visible(true)
							.with_inner_size(PhysicalSize::new(
								self.screen_width,
								self.screen_height,
							))
							.with_window_level(WindowLevel::AlwaysOnTop)
							.with_position(PhysicalPosition::new(0, 0)),
					)
					.unwrap(),
			);

			#[cfg(target_os = "windows")]
			{
				use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
				window.set_cursor_hittest(false).unwrap();

				unsafe {
					use windows::Win32::{
						Foundation::HWND,
						UI::WindowsAndMessaging::{
							SetWindowDisplayAffinity, WDA_EXCLUDEFROMCAPTURE,
						},
					};

					if let RawWindowHandle::Win32(handle) = window.window_handle().unwrap().as_raw()
					{
						let hwnd = HWND(handle.hwnd.get() as *mut _);
						SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE).ok();
					}
				}
			}

			let context = softbuffer::Context::new(window.clone()).unwrap();
			let mut surface = softbuffer::Surface::new(&context, window.clone()).unwrap();

			surface
				.resize(
					NonZeroU32::new(self.screen_width).unwrap(),
					NonZeroU32::new(self.screen_height).unwrap(),
				)
				.unwrap();

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
				if let Some(surface) = &mut self.surface {
					surface
						.resize(
							NonZeroU32::new(size.width).unwrap(),
							NonZeroU32::new(size.height).unwrap(),
						)
						.unwrap();
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
