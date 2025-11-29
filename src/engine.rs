use std::{
	borrow::Cow,
	io::BufReader,
	process::{ChildStdin, ChildStdout, Command, Stdio},
};

use anyhow::Result;
use polyglot_book_rs::PolyglotBook;

use crate::{
	chess::Score,
	config::{CONFIG, PROFILE},
	errors::{AnalysisError, Fen},
};

// TODO: Support mac and linux engine

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MoveNotation(String);

impl MoveNotation {
	pub fn new(notation: impl Into<String>) -> Self {
		Self(notation.into())
	}

	#[inline]
	pub fn as_str(&self) -> &str {
		&self.0
	}
}

impl std::fmt::Display for MoveNotation {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.0)
	}
}

impl AsRef<str> for MoveNotation {
	fn as_ref(&self) -> &str {
		&self.0
	}
}

#[derive(Debug, Clone)]
pub struct MoveWithScore {
	pub notation: MoveNotation,
	pub score: Score,
}

pub struct EngineWrapper {
	engine: ruci::Engine<BufReader<ChildStdout>, ChildStdin>,
	book: Option<PolyglotBook>,
	current_position: Option<Fen>,
}

impl EngineWrapper {
	pub fn new() -> Result<Self> {
		tracing::debug!(
			"Loading engine from {} with profile '{}'",
			CONFIG.engine.path,
			CONFIG.engine.profile
		);

		let book = if let Some(book_path) = &CONFIG.engine.book {
			tracing::debug!("Loading opening book from {}", book_path);
			Some(PolyglotBook::load(book_path).map_err(|e| {
				AnalysisError::EngineStartFailed(format!("Failed to load opening book: {}", e))
			})?)
		} else {
			None
		};

		let mut process = Command::new(&CONFIG.engine.path)
			.args(&CONFIG.engine.args)
			.stdout(Stdio::piped())
			.stdin(Stdio::piped())
			.stderr(Stdio::null())
			.spawn()
			.map_err(|e| AnalysisError::EngineStartFailed(e.to_string()))?;

		let mut engine = ruci::Engine::from_process(&mut process, false)
			.map_err(|e| AnalysisError::EngineStartFailed(e.to_string()))?;

		engine.use_uci(|_| {})?;
		engine.is_ready()?;

		engine.send(ruci::SetOption {
			name: "Threads".into(),
			value: Some(PROFILE.threads.to_string().into()),
		})?;

		engine.send(ruci::SetOption {
			name: "Hash".into(),
			value: Some(PROFILE.hash.to_string().into()),
		})?;

		engine.send(ruci::SetOption {
			name: "MultiPV".into(),
			value: Some(PROFILE.multi_pv.to_string().into()),
		})?;

		Ok(Self {
			engine,
			book,
			current_position: None,
		})
	}

	pub fn set_position(&mut self, fen: &Fen) -> Result<()> {
		let fen_str = fen.as_str();
		let parsed_fen = fen_str
			.parse()
			.map_err(|e| AnalysisError::InvalidPosition {
				fen: fen_str.to_string(),
				reason: format!("Parse error: {:?}", e),
			})?;

		self.current_position = Some(fen.clone());

		self.engine
			.send(&ruci::Position::Fen {
				fen: Cow::Owned(parsed_fen),
				moves: Cow::Borrowed(&[]),
			})
			.map_err(|e| {
				AnalysisError::InvalidPosition {
					fen: fen_str.to_string(),
					reason: e.to_string(),
				}
				.into()
			})
	}

	pub fn get_best_moves(&mut self) -> Result<Vec<MoveWithScore>> {
		if let (Some(book), Some(fen)) = (self.book.as_ref(), self.current_position.as_ref())
			&& let Some(entry) = book.get_best_move_from_fen(fen.as_str()) {
				let move_str = entry.move_string;
				tracing::info!("Opening Book Hit: {} (Weight: {})", move_str, entry.weight);

				return Ok(vec![MoveWithScore {
					notation: MoveNotation::new(move_str),
					score: Score::centipawns(0),
				}]);
			}

		let mut best_moves: Vec<MoveWithScore> = Vec::new();

		self.engine.go(
			&ruci::Go {
				depth: Some(PROFILE.depth),
				..Default::default()
			},
			|info| {
				let Some(score_with_bound) = info.score else {
					return;
				};

				let Some(first_move) = info.pv.first() else {
					return;
				};

				let score = match score_with_bound.kind {
					ruci::Score::Centipawns(cp) => Score::centipawns(cp as i32),
					ruci::Score::MateIn(moves) => Score::mate_in(moves as i8),
				};

				let notation = MoveNotation::new(first_move.to_string());

				match best_moves.iter_mut().find(|m| m.notation == notation) {
					Some(existing) if score > existing.score => {
						existing.score = score;
					}
					None => {
						best_moves.push(MoveWithScore { notation, score });
					}
					_ => {}
				}
			},
		)?;

		best_moves.truncate(PROFILE.multi_pv);

		Ok(best_moves)
	}
}
