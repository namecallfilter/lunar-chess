use std::cell::RefCell;

use font_kit::{family_name::FamilyName, font::Font, properties::Properties, source::SystemSource};
use raqote::{
	DrawOptions, DrawTarget, PathBuilder, Point, SolidSource, Source, StrokeStyle, Transform,
};

use crate::{
	chess::{BOARD_SIZE, ChessMove},
	model::detected::{DetectedBoard, DetectedPiece, Rect},
	ui::{ArrowStyle, Color, Point2D, Vec2},
};

const BOARD_OUTLINE_WIDTH: f32 = 4.0;
const GRID_LINE_WIDTH: f32 = 1.5;

const PIECE_LABEL_SIZE: f32 = 30.0;
const PIECE_LABEL_FONT_SIZE: f32 = 24.0;

const PIECE_LABEL_OFFSET_X: f32 = 5.0;
const PIECE_LABEL_OFFSET_Y: f32 = 23.0;

const COLOR_BOARD_OUTLINE: Color = Color::new(255, 0, 255, 0); // Green
const COLOR_GRID_LINES: Color = Color::new(200, 255, 255, 0); // Yellow
const COLOR_LABEL_BACKGROUND: Color = Color::new(180, 50, 50, 50); // Dark gray
const COLOR_LABEL_WHITE_PIECE: Color = Color::new(255, 255, 255, 255); // White
const COLOR_LABEL_BLACK_PIECE: Color = Color::new(255, 200, 200, 200); // Light gray

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
		Source::Solid(COLOR_BOARD_OUTLINE.to_solid_source())
	}

	pub fn grid_lines() -> Source<'static> {
		Source::Solid(COLOR_GRID_LINES.to_solid_source())
	}

	pub fn label_background() -> Source<'static> {
		Source::Solid(COLOR_LABEL_BACKGROUND.to_solid_source())
	}

	pub fn label_white_piece() -> SolidSource {
		COLOR_LABEL_WHITE_PIECE.to_solid_source()
	}

	pub fn label_black_piece() -> SolidSource {
		COLOR_LABEL_BLACK_PIECE.to_solid_source()
	}
}

pub fn draw_board_outline(dt: &mut DrawTarget, rect: &Rect) {
	let mut pb = PathBuilder::new();
	pb.rect(rect.x(), rect.y(), rect.width(), rect.height());
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

pub fn draw_chess_grid(dt: &mut DrawTarget, rect: &Rect) {
	let cell_width = rect.width() / BOARD_SIZE as f32;
	let cell_height = rect.height() / BOARD_SIZE as f32;

	let stroke_style = StrokeStyle {
		width: GRID_LINE_WIDTH,
		..Default::default()
	};

	let grid_color = cached_colors::grid_lines();

	for i in 1..BOARD_SIZE {
		let mut pb = PathBuilder::new();
		let x_pos = rect.x() + (i as f32 * cell_width);
		pb.move_to(x_pos, rect.y());
		pb.line_to(x_pos, rect.y() + rect.height());
		let path = pb.finish();

		dt.stroke(&path, &grid_color, &stroke_style, &DrawOptions::new());
	}

	for i in 1..BOARD_SIZE {
		let mut pb = PathBuilder::new();
		let y_pos = rect.y() + (i as f32 * cell_height);
		pb.move_to(rect.x(), y_pos);
		pb.line_to(rect.x() + rect.width(), y_pos);
		let path = pb.finish();

		dt.stroke(&path, &grid_color, &stroke_style, &DrawOptions::new());
	}
}

pub fn draw_move_arrow(
	dt: &mut DrawTarget, board: &DetectedBoard, chess_move: &ChessMove, color: SolidSource,
) {
	let from_file = chess_move.from.file.index();
	let from_rank = chess_move.from.rank.index();
	let to_file = chess_move.to.file.index();
	let to_rank = chess_move.to.rank.index();

	let cell_size = board.cell_size();
	let style = ArrowStyle::from_cell_size(cell_size.width, cell_size.height);

	let from = board.square_center(&chess_move.from);
	let to = board.square_center(&chess_move.to);

	let file_diff = (to_file as i8 - from_file as i8).abs();
	let rank_diff = (to_rank as i8 - from_rank as i8).abs();
	let is_knight_move = (file_diff == 2 && rank_diff == 1) || (file_diff == 1 && rank_diff == 2);

	if is_knight_move {
		let mid = Point2D::new(
			if file_diff == 2 { to.x } else { from.x },
			if rank_diff == 2 { to.y } else { from.y },
		);
		draw_knight_arrow(dt, from, mid, to, &style, color);
	} else {
		draw_straight_arrow(dt, from, to, &style, color);
	}
}

fn draw_straight_arrow(
	dt: &mut DrawTarget, from: Point2D, to: Point2D, style: &ArrowStyle, color: SolidSource,
) {
	let dx = to.x - from.x;
	let dy = to.y - from.y;
	let length = (dx * dx + dy * dy).sqrt();

	if length < 1.0 {
		return;
	}

	let norm_dx = dx / length;
	let norm_dy = dy / length;

	let shaft_start_x = from.x + norm_dx * style.margin;
	let shaft_start_y = from.y + norm_dy * style.margin;

	let shaft_end_x = to.x - norm_dx * style.head_length;
	let shaft_end_y = to.y - norm_dy * style.head_length;

	let perp_x = -norm_dy;
	let perp_y = norm_dx;

	let half_shaft = style.shaft_width / 2.0;
	let half_head = style.head_width / 2.0;

	let mut pb = PathBuilder::new();
	pb.move_to(
		shaft_start_x + perp_x * half_shaft,
		shaft_start_y + perp_y * half_shaft,
	);
	pb.line_to(
		shaft_end_x + perp_x * half_shaft,
		shaft_end_y + perp_y * half_shaft,
	);
	pb.line_to(
		shaft_end_x - perp_x * half_shaft,
		shaft_end_y - perp_y * half_shaft,
	);
	pb.line_to(
		shaft_start_x - perp_x * half_shaft,
		shaft_start_y - perp_y * half_shaft,
	);
	pb.close();

	let path = pb.finish();
	dt.fill(&path, &Source::Solid(color), &DrawOptions::new());

	let mut pb = PathBuilder::new();
	pb.move_to(to.x, to.y);
	pb.line_to(
		shaft_end_x + perp_x * half_head,
		shaft_end_y + perp_y * half_head,
	);
	pb.line_to(
		shaft_end_x - perp_x * half_head,
		shaft_end_y - perp_y * half_head,
	);
	pb.close();

	let path = pb.finish();
	dt.fill(&path, &Source::Solid(color), &DrawOptions::new());
}

fn draw_knight_arrow(
	dt: &mut DrawTarget, from: Point2D, mid: Point2D, to: Point2D, style: &ArrowStyle,
	color: SolidSource,
) {
	let dx1 = mid.x - from.x;
	let dy1 = mid.y - from.y;
	let len1 = (dx1 * dx1 + dy1 * dy1).sqrt();

	let dx2 = to.x - mid.x;
	let dy2 = to.y - mid.y;
	let len2 = (dx2 * dx2 + dy2 * dy2).sqrt();

	if len1 < 1.0 || len2 < 1.0 {
		return;
	}

	let dir1 = Vec2::new(dx1 / len1, dy1 / len1);
	let perp1 = dir1.perpendicular();

	let dir2 = Vec2::new(dx2 / len2, dy2 / len2);
	let perp2 = dir2.perpendicular();

	let seg1_start = Point2D::new(
		from.x + dir1.x * style.margin,
		from.y + dir1.y * style.margin,
	);

	let seg2_end = Point2D::new(
		to.x - dir2.x * style.head_length,
		to.y - dir2.y * style.head_length,
	);

	let half_shaft = style.shaft_width / 2.0;
	let half_head = style.head_width / 2.0;

	fn line_intersection(p1: Point2D, d1: Vec2, p2: Point2D, d2: Vec2) -> Option<Point2D> {
		let denom = d1.x * d2.y - d1.y * d2.x;
		if denom.abs() < 0.0001 {
			return None;
		}
		let t = ((p2.x - p1.x) * d2.y - (p2.y - p1.y) * d2.x) / denom;
		Some(Point2D::new(p1.x + t * d1.x, p1.y + t * d1.y))
	}

	let outer_miter = line_intersection(
		Point2D::new(
			seg1_start.x + perp1.x * half_shaft,
			seg1_start.y + perp1.y * half_shaft,
		),
		dir1,
		Point2D::new(
			seg2_end.x + perp2.x * half_shaft,
			seg2_end.y + perp2.y * half_shaft,
		),
		dir2.negate(),
	);

	let inner_miter = line_intersection(
		Point2D::new(
			seg1_start.x - perp1.x * half_shaft,
			seg1_start.y - perp1.y * half_shaft,
		),
		dir1,
		Point2D::new(
			seg2_end.x - perp2.x * half_shaft,
			seg2_end.y - perp2.y * half_shaft,
		),
		dir2.negate(),
	);

	let mut pb = PathBuilder::new();

	pb.move_to(
		seg1_start.x + perp1.x * half_shaft,
		seg1_start.y + perp1.y * half_shaft,
	);

	if let Some(m) = outer_miter {
		pb.line_to(m.x, m.y);
	} else {
		pb.line_to(mid.x + perp1.x * half_shaft, mid.y + perp1.y * half_shaft);
	}

	pb.line_to(
		seg2_end.x + perp2.x * half_shaft,
		seg2_end.y + perp2.y * half_shaft,
	);

	pb.line_to(
		seg2_end.x - perp2.x * half_shaft,
		seg2_end.y - perp2.y * half_shaft,
	);

	if let Some(m) = inner_miter {
		pb.line_to(m.x, m.y);
	} else {
		pb.line_to(mid.x - perp1.x * half_shaft, mid.y - perp1.y * half_shaft);
	}

	pb.line_to(
		seg1_start.x - perp1.x * half_shaft,
		seg1_start.y - perp1.y * half_shaft,
	);

	pb.close();

	let path = pb.finish();
	dt.fill(&path, &Source::Solid(color), &DrawOptions::new());

	let mut pb = PathBuilder::new();
	pb.move_to(to.x, to.y);
	pb.line_to(
		seg2_end.x + perp2.x * half_head,
		seg2_end.y + perp2.y * half_head,
	);
	pb.line_to(
		seg2_end.x - perp2.x * half_head,
		seg2_end.y - perp2.y * half_head,
	);
	pb.close();

	let path = pb.finish();
	dt.fill(&path, &Source::Solid(color), &DrawOptions::new());
}

pub fn draw_piece_labels(dt: &mut DrawTarget, pieces: &[DetectedPiece]) {
	LABEL_FONT.with(|font_cell| {
		let font_ref = font_cell.borrow();
		let Some(font) = font_ref.as_ref() else {
			static WARNED: std::sync::atomic::AtomicBool =
				std::sync::atomic::AtomicBool::new(false);
			if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
				tracing::warn!("System font for labels not found, skipping label rendering");
			}
			return;
		};

		for piece in pieces {
			let mut pb = PathBuilder::new();
			pb.rect(piece.x(), piece.y(), PIECE_LABEL_SIZE, PIECE_LABEL_SIZE);
			let path = pb.finish();

			dt.fill(
				&path,
				&cached_colors::label_background(),
				&DrawOptions::new(),
			);

			let text = piece.piece_type.to_fen_char().to_string();
			let color = if piece.piece_type.is_white() {
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
					piece.x() + PIECE_LABEL_OFFSET_X,
					piece.y() + PIECE_LABEL_OFFSET_Y,
				),
				&Source::Solid(color),
				&DrawOptions::new(),
			);
		}

		dt.set_transform(&Transform::identity());
	});
}
