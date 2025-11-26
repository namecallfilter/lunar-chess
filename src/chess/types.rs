use super::score::Score;

pub const BOARD_SIZE: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct File(u8);

impl File {
	pub fn new(value: u8) -> Option<Self> {
		if (value as usize) < BOARD_SIZE {
			Some(Self(value))
		} else {
			None
		}
	}

	#[inline]
	pub fn index(self) -> u8 {
		self.0
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rank(u8);

impl Rank {
	pub fn new(value: u8) -> Option<Self> {
		if (value as usize) < BOARD_SIZE {
			Some(Self(value))
		} else {
			None
		}
	}

	#[inline]
	pub fn index(self) -> u8 {
		self.0
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Square {
	pub file: File,
	pub rank: Rank,
}

impl Square {
	pub fn new(file: File, rank: Rank) -> Self {
		Self { file, rank }
	}

	pub fn from_indices(file: u8, rank: u8) -> Option<Self> {
		Some(Self {
			file: File::new(file)?,
			rank: Rank::new(rank)?,
		})
	}
}

#[derive(Debug, Clone, Copy)]
pub struct ChessMove {
	pub from: Square,
	pub to: Square,
	pub score: Score,
}

impl ChessMove {
	pub fn new(from: Square, to: Square) -> Self {
		Self {
			from,
			to,
			score: Score::default(),
		}
	}

	pub fn with_score(from: Square, to: Square, score: Score) -> Self {
		Self { from, to, score }
	}
}
