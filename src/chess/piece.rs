#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceType {
	WhiteKing,
	WhiteQueen,
	WhiteRook,
	WhiteBishop,
	WhiteKnight,
	WhitePawn,
	BlackKing,
	BlackQueen,
	BlackRook,
	BlackBishop,
	BlackKnight,
	BlackPawn,
}

impl PieceType {
	#[inline]
	pub const fn to_fen_char(self) -> char {
		match self {
			Self::WhiteKing => 'K',
			Self::WhiteQueen => 'Q',
			Self::WhiteRook => 'R',
			Self::WhiteBishop => 'B',
			Self::WhiteKnight => 'N',
			Self::WhitePawn => 'P',
			Self::BlackKing => 'k',
			Self::BlackQueen => 'q',
			Self::BlackRook => 'r',
			Self::BlackBishop => 'b',
			Self::BlackKnight => 'n',
			Self::BlackPawn => 'p',
		}
	}

	pub const fn from_class_index(index: usize) -> Option<Self> {
		match index {
			0 => Some(Self::BlackRook),
			1 => Some(Self::BlackKnight),
			2 => Some(Self::BlackBishop),
			3 => Some(Self::BlackQueen),
			4 => Some(Self::BlackKing),
			5 => Some(Self::BlackPawn),
			6 => Some(Self::WhiteRook),
			7 => Some(Self::WhiteKnight),
			8 => Some(Self::WhiteBishop),
			9 => Some(Self::WhiteQueen),
			10 => Some(Self::WhiteKing),
			11 => Some(Self::WhitePawn),
			_ => None,
		}
	}

	#[inline]
	pub const fn is_white(self) -> bool {
		matches!(
			self,
			Self::WhiteKing
				| Self::WhiteQueen
				| Self::WhiteRook
				| Self::WhiteBishop
				| Self::WhiteKnight
				| Self::WhitePawn
		)
	}
}

impl std::fmt::Display for PieceType {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.to_fen_char())
	}
}
