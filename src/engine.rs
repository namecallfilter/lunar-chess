use std::{
	borrow::Cow,
	io::BufReader,
	process::{ChildStdin, ChildStdout, Command, Stdio},
};

use anyhow::Result;

use crate::{
	chess::Score,
	config::{CONFIG, PROFILE},
	errors::AnalysisError,
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
}

impl EngineWrapper {
	pub fn new() -> Result<Self> {
		tracing::debug!(
			"Loading engine from {} with profile '{}'",
			CONFIG.engine.path,
			CONFIG.engine.profile
		);

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

		Ok(Self { engine })
	}

	pub fn set_position(&mut self, fen: &str) -> Result<()> {
		let parsed_fen = fen.parse().map_err(|e| AnalysisError::InvalidPosition {
			fen: fen.to_string(),
			reason: format!("Parse error: {:?}", e),
		})?;

		self.engine
			.send(&ruci::Position::Fen {
				fen: Cow::Owned(parsed_fen),
				moves: Cow::Borrowed(&[]),
			})
			.map_err(|e| {
				AnalysisError::InvalidPosition {
					fen: fen.to_string(),
					reason: e.to_string(),
				}
				.into()
			})
	}

	pub fn get_best_moves(&mut self) -> Result<Vec<MoveWithScore>> {
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
