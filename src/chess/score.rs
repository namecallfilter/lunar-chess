#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Score {
	Centipawns(i32),
	MateIn(i8),
}

impl Score {
	const MATE_SCORE_BASE: i32 = 10000;

	#[inline]
	pub const fn centipawns(cp: i32) -> Self {
		Self::Centipawns(cp)
	}

	#[inline]
	pub const fn mate_in(moves: i8) -> Self {
		Self::MateIn(moves)
	}

	/// to_numeric encodes MateIn(n) such that positive n denotes mate-for-side-X (document which side),
	/// and returns a large magnitude value so mate beats any centipawn score.
	#[inline]
	pub const fn to_numeric(self) -> i32 {
		match self {
			Self::Centipawns(cp) => cp,
			Self::MateIn(moves) => {
				let sign = if moves > 0 { 1 } else { -1 };
				sign * (Self::MATE_SCORE_BASE - (moves.abs() as i32))
			}
		}
	}
}

impl Default for Score {
	fn default() -> Self {
		Self::Centipawns(0)
	}
}

impl PartialOrd for Score {
	fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		Some(self.cmp(other))
	}
}

impl Ord for Score {
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		self.to_numeric().cmp(&other.to_numeric())
	}
}

impl std::fmt::Display for Score {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Centipawns(cp) => {
				let pawns = *cp as f32 / 100.0;
				if pawns >= 0.0 {
					write!(f, "+{:.2}", pawns)
				} else {
					write!(f, "{:.2}", pawns)
				}
			}
			Self::MateIn(moves) => {
				if *moves > 0 {
					write!(f, "M{}", moves)
				} else {
					write!(f, "-M{}", moves.abs())
				}
			}
		}
	}
}
