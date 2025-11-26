#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PlayerColor {
	#[default]
	White,
	Black,
}

impl PlayerColor {
	#[inline]
	pub const fn is_white(self) -> bool {
		matches!(self, Self::White)
	}

	#[inline]
	pub const fn is_black(self) -> bool {
		matches!(self, Self::Black)
	}

	#[inline]
	pub const fn to_fen_char(self) -> char {
		match self {
			Self::White => 'w',
			Self::Black => 'b',
		}
	}
}

impl std::fmt::Display for PlayerColor {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::White => write!(f, "white"),
			Self::Black => write!(f, "black"),
		}
	}
}
