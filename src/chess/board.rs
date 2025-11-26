use super::{
	color::PlayerColor,
	types::{BOARD_SIZE, ChessMove, File, Rank, Square},
};

pub fn parse_move(move_str: &str, player_color: PlayerColor) -> Option<ChessMove> {
	let bytes = move_str.as_bytes();
	if bytes.len() < 4 {
		return None;
	}

	let from_file = bytes[0].checked_sub(b'a')?;
	let from_rank_chess = bytes[1].checked_sub(b'1')?;
	let to_file = bytes[2].checked_sub(b'a')?;
	let to_rank_chess = bytes[3].checked_sub(b'1')?;

	let board_size = BOARD_SIZE as u8;
	if from_file >= board_size
		|| from_rank_chess >= board_size
		|| to_file >= board_size
		|| to_rank_chess >= board_size
	{
		return None;
	}

	let (from_rank, to_rank) = if player_color.is_white() {
		(
			board_size - 1 - from_rank_chess,
			board_size - 1 - to_rank_chess,
		)
	} else {
		(from_rank_chess, to_rank_chess)
	};

	let (from_file_final, to_file_final) = if player_color.is_white() {
		(from_file, to_file)
	} else {
		(board_size - 1 - from_file, board_size - 1 - to_file)
	};

	Some(ChessMove::new(
		Square::new(File::new(from_file_final)?, Rank::new(from_rank)?),
		Square::new(File::new(to_file_final)?, Rank::new(to_rank)?),
	))
}
