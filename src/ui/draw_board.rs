use std::cell::RefCell;

use font_kit::{family_name::FamilyName, font::Font, properties::Properties, source::SystemSource};
use raqote::{
	DrawOptions, DrawTarget, PathBuilder, Point, SolidSource, Source, StrokeStyle, Transform,
};

use crate::model::detected::{DetectedBoard, DetectedPiece};

const BOARD_OUTLINE_WIDTH: f32 = 4.0;
const GRID_LINE_WIDTH: f32 = 1.5;

const ARROW_SHAFT_WIDTH_FRACTION: f32 = 0.25;
const ARROW_HEAD_LENGTH_FRACTION: f32 = 0.5;
const ARROW_HEAD_WIDTH_MULTIPLIER: f32 = 2.5;
const ARROW_MARGIN_FRACTION: f32 = 0.3;

const PIECE_LABEL_SIZE: f32 = 30.0;
const PIECE_LABEL_FONT_SIZE: f32 = 24.0;

const PIECE_LABEL_OFFSET_X: f32 = 5.0;
const PIECE_LABEL_OFFSET_Y: f32 = 23.0;

const COLOR_BOARD_OUTLINE: (u8, u8, u8, u8) = (255, 0, 255, 0); // Green
const COLOR_GRID_LINES: (u8, u8, u8, u8) = (200, 255, 255, 0); // Yellow
const COLOR_LABEL_BACKGROUND: (u8, u8, u8, u8) = (180, 50, 50, 50); // Dark gray
const COLOR_LABEL_WHITE_PIECE: (u8, u8, u8, u8) = (255, 255, 255, 255); // White
const COLOR_LABEL_BLACK_PIECE: (u8, u8, u8, u8) = (255, 200, 200, 200); // Light gray

thread_local! {
	static LABEL_FONT: RefCell<Option<Font>> = RefCell::new(
		SystemSource::new()
			.select_best_match(
				&[FamilyName::Title("Arial".into()), FamilyName::SansSerif],
				&Properties::new(),
			)
			.ok()
			.and_then(|handle| handle.load().ok())
	);
}

mod cached_colors {
	use super::*;

	pub fn board_outline() -> Source<'static> {
		Source::Solid(SolidSource::from_unpremultiplied_argb(
			COLOR_BOARD_OUTLINE.0,
			COLOR_BOARD_OUTLINE.1,
			COLOR_BOARD_OUTLINE.2,
			COLOR_BOARD_OUTLINE.3,
		))
	}

	pub fn grid_lines() -> Source<'static> {
		Source::Solid(SolidSource::from_unpremultiplied_argb(
			COLOR_GRID_LINES.0,
			COLOR_GRID_LINES.1,
			COLOR_GRID_LINES.2,
			COLOR_GRID_LINES.3,
		))
	}

	pub fn label_background() -> Source<'static> {
		Source::Solid(SolidSource::from_unpremultiplied_argb(
			COLOR_LABEL_BACKGROUND.0,
			COLOR_LABEL_BACKGROUND.1,
			COLOR_LABEL_BACKGROUND.2,
			COLOR_LABEL_BACKGROUND.3,
		))
	}

	pub fn label_white_piece() -> SolidSource {
		SolidSource::from_unpremultiplied_argb(
			COLOR_LABEL_WHITE_PIECE.0,
			COLOR_LABEL_WHITE_PIECE.1,
			COLOR_LABEL_WHITE_PIECE.2,
			COLOR_LABEL_WHITE_PIECE.3,
		)
	}

	pub fn label_black_piece() -> SolidSource {
		SolidSource::from_unpremultiplied_argb(
			COLOR_LABEL_BLACK_PIECE.0,
			COLOR_LABEL_BLACK_PIECE.1,
			COLOR_LABEL_BLACK_PIECE.2,
			COLOR_LABEL_BLACK_PIECE.3,
		)
	}
}

pub fn draw_board_outline(dt: &mut DrawTarget, x: f32, y: f32, width: f32, height: f32) {
	let mut pb = PathBuilder::new();
	pb.rect(x, y, width, height);
	let path = pb.finish();

	let stroke_style = StrokeStyle {
		width: BOARD_OUTLINE_WIDTH,
		..Default::default()
	};

	dt.stroke(
		&path,
		&cached_colors::board_outline(),
		&stroke_style,
		&DrawOptions::new(),
	);
}

pub fn draw_chess_grid(dt: &mut DrawTarget, x: f32, y: f32, width: f32, height: f32) {
	let cell_width = width / 8.0;
	let cell_height = height / 8.0;

	let stroke_style = StrokeStyle {
		width: GRID_LINE_WIDTH,
		..Default::default()
	};

	let grid_color = cached_colors::grid_lines();

	for i in 1..8 {
		let mut pb = PathBuilder::new();
		let x_pos = x + (i as f32 * cell_width);
		pb.move_to(x_pos, y);
		pb.line_to(x_pos, y + height);
		let path = pb.finish();

		dt.stroke(&path, &grid_color, &stroke_style, &DrawOptions::new());
	}

	for i in 1..8 {
		let mut pb = PathBuilder::new();
		let y_pos = y + (i as f32 * cell_height);
		pb.move_to(x, y_pos);
		pb.line_to(x + width, y_pos);
		let path = pb.finish();

		dt.stroke(&path, &grid_color, &stroke_style, &DrawOptions::new());
	}
}

pub fn draw_move_arrow(
	dt: &mut DrawTarget, board: &DetectedBoard, from_file: u8, from_rank: u8, to_file: u8,
	to_rank: u8, color: (u8, u8, u8, u8),
) {
	if from_file >= 8 || from_rank >= 8 || to_file >= 8 || to_rank >= 8 {
		return;
	}

	let (cell_width, cell_height) = board.cell_size();

	let from_x = board.x + (from_file as f32 + 0.5) * cell_width;
	let from_y = board.y + (from_rank as f32 + 0.5) * cell_height;
	let to_x = board.x + (to_file as f32 + 0.5) * cell_width;
	let to_y = board.y + (to_rank as f32 + 0.5) * cell_height;

	let file_diff = (to_file as i8 - from_file as i8).abs();
	let rank_diff = (to_rank as i8 - from_rank as i8).abs();
	let is_knight_move = (file_diff == 2 && rank_diff == 1) || (file_diff == 1 && rank_diff == 2);

	if is_knight_move {
		draw_knight_arrow(dt, board, from_file, from_rank, to_file, to_rank, color);
	} else {
		draw_straight_arrow(
			dt,
			from_x,
			from_y,
			to_x,
			to_y,
			cell_width,
			cell_height,
			color,
		);
	}
}

#[allow(clippy::too_many_arguments)]
fn draw_straight_arrow(
	dt: &mut DrawTarget, from_x: f32, from_y: f32, to_x: f32, to_y: f32, cell_width: f32,
	cell_height: f32, color: (u8, u8, u8, u8),
) {
	let dx = to_x - from_x;
	let dy = to_y - from_y;
	let length = (dx * dx + dy * dy).sqrt();

	if length < 1.0 {
		return;
	}

	let norm_dx = dx / length;
	let norm_dy = dy / length;

	let arrow_width = cell_width.min(cell_height) * ARROW_SHAFT_WIDTH_FRACTION;
	let arrow_head_length = cell_width.min(cell_height) * ARROW_HEAD_LENGTH_FRACTION;
	let arrow_head_width = arrow_width * ARROW_HEAD_WIDTH_MULTIPLIER;

	let margin = cell_width.min(cell_height) * ARROW_MARGIN_FRACTION;
	let shaft_start_x = from_x + norm_dx * margin;
	let shaft_start_y = from_y + norm_dy * margin;

	let shaft_end_x = to_x - norm_dx * arrow_head_length;
	let shaft_end_y = to_y - norm_dy * arrow_head_length;

	let perp_x = -norm_dy;
	let perp_y = norm_dx;

	let mut pb = PathBuilder::new();
	pb.move_to(
		shaft_start_x + perp_x * arrow_width / 2.0,
		shaft_start_y + perp_y * arrow_width / 2.0,
	);
	pb.line_to(
		shaft_end_x + perp_x * arrow_width / 2.0,
		shaft_end_y + perp_y * arrow_width / 2.0,
	);
	pb.line_to(
		shaft_end_x - perp_x * arrow_width / 2.0,
		shaft_end_y - perp_y * arrow_width / 2.0,
	);
	pb.line_to(
		shaft_start_x - perp_x * arrow_width / 2.0,
		shaft_start_y - perp_y * arrow_width / 2.0,
	);
	pb.close();

	let path = pb.finish();
	dt.fill(
		&path,
		&Source::Solid(SolidSource::from_unpremultiplied_argb(
			color.0, color.1, color.2, color.3,
		)),
		&DrawOptions::new(),
	);

	let mut pb = PathBuilder::new();
	pb.move_to(to_x, to_y);
	pb.line_to(
		shaft_end_x + perp_x * arrow_head_width / 2.0,
		shaft_end_y + perp_y * arrow_head_width / 2.0,
	);
	pb.line_to(
		shaft_end_x - perp_x * arrow_head_width / 2.0,
		shaft_end_y - perp_y * arrow_head_width / 2.0,
	);
	pb.close();

	let path = pb.finish();
	dt.fill(
		&path,
		&Source::Solid(SolidSource::from_unpremultiplied_argb(
			color.0, color.1, color.2, color.3,
		)),
		&DrawOptions::new(),
	);
}

#[allow(clippy::too_many_arguments)]
fn draw_knight_arrow(
	dt: &mut DrawTarget, board: &DetectedBoard, from_file: u8, from_rank: u8, to_file: u8,
	to_rank: u8, color: (u8, u8, u8, u8),
) {
	let (cell_width, cell_height) = board.cell_size();

	let from_x = board.x + (from_file as f32 + 0.5) * cell_width;
	let from_y = board.y + (from_rank as f32 + 0.5) * cell_height;
	let to_x = board.x + (to_file as f32 + 0.5) * cell_width;
	let to_y = board.y + (to_rank as f32 + 0.5) * cell_height;

	let file_diff = (to_file as i8 - from_file as i8).abs();
	let rank_diff = (to_rank as i8 - from_rank as i8).abs();

	let mid_x = if file_diff == 2 { to_x } else { from_x };

	let mid_y = if rank_diff == 2 { to_y } else { from_y };

	let arrow_width = cell_width.min(cell_height) * ARROW_SHAFT_WIDTH_FRACTION;
	let arrow_head_length = cell_width.min(cell_height) * ARROW_HEAD_LENGTH_FRACTION;
	let arrow_head_width = arrow_width * ARROW_HEAD_WIDTH_MULTIPLIER;
	let margin = cell_width.min(cell_height) * ARROW_MARGIN_FRACTION;

	let dx1 = mid_x - from_x;
	let dy1 = mid_y - from_y;
	let len1 = (dx1 * dx1 + dy1 * dy1).sqrt();

	let dx2 = to_x - mid_x;
	let dy2 = to_y - mid_y;
	let len2 = (dx2 * dx2 + dy2 * dy2).sqrt();

	if len1 < 1.0 || len2 < 1.0 {
		return;
	}

	let norm_dx1 = dx1 / len1;
	let norm_dy1 = dy1 / len1;
	let perp_x1 = -norm_dy1;
	let perp_y1 = norm_dx1;

	let norm_dx2 = dx2 / len2;
	let norm_dy2 = dy2 / len2;
	let perp_x2 = -norm_dy2;
	let perp_y2 = norm_dx2;

	let seg1_start_x = from_x + norm_dx1 * margin;
	let seg1_start_y = from_y + norm_dy1 * margin;

	let seg2_end_x = to_x - norm_dx2 * arrow_head_length;
	let seg2_end_y = to_y - norm_dy2 * arrow_head_length;

	fn line_intersection(
		x1: f32, y1: f32, dx1: f32, dy1: f32, x2: f32, y2: f32, dx2: f32, dy2: f32,
	) -> Option<(f32, f32)> {
		let denom = dx1 * dy2 - dy1 * dx2;
		if denom.abs() < 0.0001 {
			return None;
		}
		let t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / denom;
		Some((x1 + t * dx1, y1 + t * dy1))
	}

	let outer_miter = line_intersection(
		seg1_start_x + perp_x1 * arrow_width / 2.0,
		seg1_start_y + perp_y1 * arrow_width / 2.0,
		norm_dx1,
		norm_dy1,
		seg2_end_x + perp_x2 * arrow_width / 2.0,
		seg2_end_y + perp_y2 * arrow_width / 2.0,
		-norm_dx2,
		-norm_dy2,
	);

	let inner_miter = line_intersection(
		seg1_start_x - perp_x1 * arrow_width / 2.0,
		seg1_start_y - perp_y1 * arrow_width / 2.0,
		norm_dx1,
		norm_dy1,
		seg2_end_x - perp_x2 * arrow_width / 2.0,
		seg2_end_y - perp_y2 * arrow_width / 2.0,
		-norm_dx2,
		-norm_dy2,
	);

	let mut pb = PathBuilder::new();

	pb.move_to(
		seg1_start_x + perp_x1 * arrow_width / 2.0,
		seg1_start_y + perp_y1 * arrow_width / 2.0,
	);

	if let Some((mx, my)) = outer_miter {
		pb.line_to(mx, my);
	} else {
		pb.line_to(
			mid_x + perp_x1 * arrow_width / 2.0,
			mid_y + perp_y1 * arrow_width / 2.0,
		);
	}

	pb.line_to(
		seg2_end_x + perp_x2 * arrow_width / 2.0,
		seg2_end_y + perp_y2 * arrow_width / 2.0,
	);

	pb.line_to(
		seg2_end_x - perp_x2 * arrow_width / 2.0,
		seg2_end_y - perp_y2 * arrow_width / 2.0,
	);

	if let Some((mx, my)) = inner_miter {
		pb.line_to(mx, my);
	} else {
		pb.line_to(
			mid_x - perp_x1 * arrow_width / 2.0,
			mid_y - perp_y1 * arrow_width / 2.0,
		);
	}

	pb.line_to(
		seg1_start_x - perp_x1 * arrow_width / 2.0,
		seg1_start_y - perp_y1 * arrow_width / 2.0,
	);

	pb.close();

	let path = pb.finish();
	dt.fill(
		&path,
		&Source::Solid(SolidSource::from_unpremultiplied_argb(
			color.0, color.1, color.2, color.3,
		)),
		&DrawOptions::new(),
	);

	let mut pb = PathBuilder::new();
	pb.move_to(to_x, to_y);
	pb.line_to(
		seg2_end_x + perp_x2 * arrow_head_width / 2.0,
		seg2_end_y + perp_y2 * arrow_head_width / 2.0,
	);
	pb.line_to(
		seg2_end_x - perp_x2 * arrow_head_width / 2.0,
		seg2_end_y - perp_y2 * arrow_head_width / 2.0,
	);
	pb.close();

	let path = pb.finish();
	dt.fill(
		&path,
		&Source::Solid(SolidSource::from_unpremultiplied_argb(
			color.0, color.1, color.2, color.3,
		)),
		&DrawOptions::new(),
	);
}

pub fn draw_piece_labels(dt: &mut DrawTarget, pieces: &[DetectedPiece]) {
	LABEL_FONT.with(|font_cell| {
		let font_ref = font_cell.borrow();
		let Some(font) = font_ref.as_ref() else {
			return;
		};

		for piece in pieces {
			let mut pb = PathBuilder::new();
			pb.rect(piece.x, piece.y, PIECE_LABEL_SIZE, PIECE_LABEL_SIZE);
			let path = pb.finish();

			dt.fill(
				&path,
				&cached_colors::label_background(),
				&DrawOptions::new(),
			);

			let text = piece.piece_type.to_string();
			let color = if piece.piece_type.is_uppercase() {
				cached_colors::label_white_piece()
			} else {
				cached_colors::label_black_piece()
			};

			dt.set_transform(&Transform::identity());
			dt.draw_text(
				font,
				PIECE_LABEL_FONT_SIZE,
				&text,
				Point::new(
					piece.x + PIECE_LABEL_OFFSET_X,
					piece.y + PIECE_LABEL_OFFSET_Y,
				),
				&Source::Solid(color),
				&DrawOptions::new(),
			);
		}

		dt.set_transform(&Transform::identity());
	});
}
