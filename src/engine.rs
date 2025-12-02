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
	_process: std::process::Child,
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

		for (name, value) in &PROFILE.uci {
			engine.send(ruci::SetOption {
				name: name.clone().into(),
				value: Some(value.clone().into()),
			})?;
		}

		Ok(Self {
			engine,
			_process: process,
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

	pub fn get_best_moves<F>(&mut self, mut on_update: F) -> Result<Vec<MoveWithScore>>
	where
		F: FnMut(&[MoveWithScore]),
	{
		if let (Some(book), Some(fen)) = (self.book.as_ref(), self.current_position.as_ref())
			&& let Some(entry) = book.get_best_move_from_fen(fen.as_str())
		{
			let move_str = entry.move_string;
			tracing::info!("Opening Book Hit: {} (Weight: {})", move_str, entry.weight);

			let moves = vec![MoveWithScore {
				notation: MoveNotation::new(move_str),
				score: Score::centipawns(0),
			}];

			on_update(&moves);
			return Ok(moves);
		}

		let mut best_moves: Vec<MoveWithScore> = Vec::new();

		let go_params = if let Some(config) = &PROFILE.go {
			ruci::Go {
				search_moves: if let Some(moves) = &config.search_moves {
					let uci_moves: Result<Vec<_>, _> = moves.iter().map(|m| m.parse()).collect();
					match uci_moves {
						Ok(m) => Cow::Owned(m),
						Err(e) => {
							tracing::warn!("Failed to parse search moves: {}", e);
							Cow::Borrowed(&[])
						}
					}
				} else {
					Cow::Borrowed(&[])
				},
				ponder: config.ponder.unwrap_or(false),
				w_time: config.wtime,
				b_time: config.btime,
				w_inc: config.winc.and_then(std::num::NonZeroUsize::new),
				b_inc: config.binc.and_then(std::num::NonZeroUsize::new),
				moves_to_go: config.movestogo.and_then(std::num::NonZeroUsize::new),
				depth: config.depth,
				nodes: config.nodes,
				mate: config.mate,
				move_time: config.movetime,
				infinite: config.infinite.unwrap_or(false),
			}
		} else {
			ruci::Go::default()
		};

		let multi_pv = PROFILE
			.uci
			.get("MultiPV")
			.and_then(|v| v.parse::<usize>().ok())
			.unwrap_or(1);

		self.engine.go(&go_params, |info| {
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
				Some(existing) => {
					existing.score = score;
				}
				None => {
					best_moves.push(MoveWithScore { notation, score });
				}
			}

			best_moves.truncate(multi_pv);
			on_update(&best_moves);
		})?;

		Ok(best_moves)
	}
}
