use super::{
	color::PlayerColor,
	types::{BOARD_SIZE, ChessMove, File, Rank, Square},
};

pub fn parse_move(move_str: &str, player_color: PlayerColor) -> Option<ChessMove> {
	let s = move_str.trim();
	if s.len() != 4 {
		return None;
	}
	let bytes = s.as_bytes();

	let from_file = bytes[0].checked_sub(b'a')?;
	let from_rank_char = bytes[1].checked_sub(b'1')?;
	let to_file = bytes[2].checked_sub(b'a')?;
	let to_rank_char = bytes[3].checked_sub(b'1')?;

	let board_size = BOARD_SIZE as u8;
	if from_file >= board_size
		|| from_rank_char >= board_size
		|| to_file >= board_size
		|| to_rank_char >= board_size
	{
		return None;
	}

	let (from_file_final, from_rank_final) = if player_color.is_white() {
		(from_file, board_size - 1 - from_rank_char)
	} else {
		(board_size - 1 - from_file, from_rank_char)
	};

	let (to_file_final, to_rank_final) = if player_color.is_white() {
		(to_file, board_size - 1 - to_rank_char)
	} else {
		(board_size - 1 - to_file, to_rank_char)
	};

	Some(ChessMove::new(
		Square::new(File::new(from_file_final)?, Rank::new(from_rank_final)?),
		Square::new(File::new(to_file_final)?, Rank::new(to_rank_final)?),
	))
}
