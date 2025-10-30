use crate::drawing::{BestMove, DetectedBoard, DetectedPiece};

pub fn to_fen(board: &DetectedBoard, pieces: &[DetectedPiece]) -> String {
	let mut chess_board = [[None; 8]; 8];

	for piece in pieces {
		if let Some((file, rank)) = piece_to_square(piece, board)
			&& file < 8
			&& rank < 8
		{
			if chess_board[rank as usize][file as usize].is_some() {
				tracing::warn!(
					"Duplicate piece detected at ({}, {}), keeping first one",
					file,
					rank
				);
				continue;
			}
			chess_board[rank as usize][file as usize] = Some(piece.piece_type);
		}
	}

	let mut white_bottom = 0;
	let mut white_top = 0;

	for (rank_idx, row) in chess_board.iter().enumerate() {
		for piece in row.iter().flatten() {
			if piece.is_uppercase() {
				if rank_idx < 4 {
					white_top += 1;
				} else {
					white_bottom += 1;
				}
			}
		}
	}

	let active_color = if white_bottom > white_top { "w" } else { "b" };

	let mut fen = String::new();

	for (rank, row) in chess_board.iter().enumerate() {
		let mut empty_count = 0;

		for cell in row.iter() {
			match *cell {
				Some(piece) => {
					if empty_count > 0 {
						fen.push_str(&empty_count.to_string());
						empty_count = 0;
					}
					fen.push(piece);
				}
				None => {
					empty_count += 1;
				}
			}
		}

		if empty_count > 0 {
			fen.push_str(&empty_count.to_string());
		}

		if rank < 7 {
			fen.push('/');
		}
	}

	fen.push_str(&format!(" {} - - 0 1", active_color));

	tracing::debug!("Generated FEN: {}", fen);
	fen
}

pub fn piece_to_square(piece: &DetectedPiece, board: &DetectedBoard) -> Option<(u8, u8)> {
	let cell_width = board.width / 8.0;
	let cell_height = board.height / 8.0;

	let piece_center_x = piece.x + (piece.width / 2.0);
	let piece_center_y = piece.y + (piece.height / 2.0);

	let rel_x = piece_center_x - board.x;
	let rel_y = piece_center_y - board.y;

	let file = (rel_x / cell_width) as i32;
	let rank = (rel_y / cell_height) as i32;

	if (0..8).contains(&file) && (0..8).contains(&rank) {
		Some((file as u8, rank as u8))
	} else {
		None
	}
}

pub fn parse_move(move_str: &str) -> Option<BestMove> {
	let bytes = move_str.as_bytes();
	if bytes.len() < 4 {
		return None;
	}

	let from_file = bytes[0].checked_sub(b'a')?;
	let from_rank_chess = bytes[1].checked_sub(b'1')?;
	let to_file = bytes[2].checked_sub(b'a')?;
	let to_rank_chess = bytes[3].checked_sub(b'1')?;

	if from_file >= 8 || from_rank_chess >= 8 || to_file >= 8 || to_rank_chess >= 8 {
		return None;
	}

	let from_rank = 7 - from_rank_chess;
	let to_rank = 7 - to_rank_chess;

	Some(BestMove {
		from_file,
		from_rank,
		to_file,
		to_rank,
		score: 0,
	})
}
