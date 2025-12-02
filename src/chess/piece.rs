use super::color::PlayerColor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceKind {
	King,
	Queen,
	Rook,
	Bishop,
	Knight,
	Pawn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PieceType {
	color: PlayerColor,
	kind: PieceKind,
}

impl PieceType {
	pub const BLACK_BISHOP: Self = Self::new(PlayerColor::Black, PieceKind::Bishop);
	pub const BLACK_KING: Self = Self::new(PlayerColor::Black, PieceKind::King);
	pub const BLACK_KNIGHT: Self = Self::new(PlayerColor::Black, PieceKind::Knight);
	pub const BLACK_PAWN: Self = Self::new(PlayerColor::Black, PieceKind::Pawn);
	pub const BLACK_QUEEN: Self = Self::new(PlayerColor::Black, PieceKind::Queen);
	pub const BLACK_ROOK: Self = Self::new(PlayerColor::Black, PieceKind::Rook);
	pub const WHITE_BISHOP: Self = Self::new(PlayerColor::White, PieceKind::Bishop);
	pub const WHITE_KING: Self = Self::new(PlayerColor::White, PieceKind::King);
	pub const WHITE_KNIGHT: Self = Self::new(PlayerColor::White, PieceKind::Knight);
	pub const WHITE_PAWN: Self = Self::new(PlayerColor::White, PieceKind::Pawn);
	pub const WHITE_QUEEN: Self = Self::new(PlayerColor::White, PieceKind::Queen);
	pub const WHITE_ROOK: Self = Self::new(PlayerColor::White, PieceKind::Rook);

	pub const fn new(color: PlayerColor, kind: PieceKind) -> Self {
		Self { color, kind }
	}

	#[inline]
	pub const fn to_fen_char(self) -> char {
		match (self.color, self.kind) {
			(PlayerColor::White, PieceKind::King) => 'K',
			(PlayerColor::White, PieceKind::Queen) => 'Q',
			(PlayerColor::White, PieceKind::Rook) => 'R',
			(PlayerColor::White, PieceKind::Bishop) => 'B',
			(PlayerColor::White, PieceKind::Knight) => 'N',
			(PlayerColor::White, PieceKind::Pawn) => 'P',
			(PlayerColor::Black, PieceKind::King) => 'k',
			(PlayerColor::Black, PieceKind::Queen) => 'q',
			(PlayerColor::Black, PieceKind::Rook) => 'r',
			(PlayerColor::Black, PieceKind::Bishop) => 'b',
			(PlayerColor::Black, PieceKind::Knight) => 'n',
			(PlayerColor::Black, PieceKind::Pawn) => 'p',
		}
	}

	pub const fn from_class_index(index: usize) -> Option<Self> {
		match index {
			0 => Some(Self::BLACK_ROOK),
			1 => Some(Self::BLACK_KNIGHT),
			2 => Some(Self::BLACK_BISHOP),
			3 => Some(Self::BLACK_QUEEN),
			4 => Some(Self::BLACK_KING),
			5 => Some(Self::BLACK_PAWN),
			6 => Some(Self::WHITE_ROOK),
			7 => Some(Self::WHITE_KNIGHT),
			8 => Some(Self::WHITE_BISHOP),
			9 => Some(Self::WHITE_QUEEN),
			10 => Some(Self::WHITE_KING),
			11 => Some(Self::WHITE_PAWN),
			_ => None,
		}
	}

	#[inline]
	pub const fn is_white(self) -> bool {
		matches!(self.color, PlayerColor::White)
	}
}

impl std::fmt::Display for PieceType {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.to_fen_char())
	}
}
