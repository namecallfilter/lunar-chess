use std::{
	borrow::Cow,
	io::BufReader,
	process::{ChildStdin, ChildStdout, Command, Stdio},
};

use anyhow::Result;

use crate::{config::CONFIG, errors::AnalysisError};

const ENGINE_PATH: &str = "models/chessbots/lc0.exe";
const DEFAULT_WEIGHTS_PATH: &str = "models/chessbots/maia-1100.pb.gz";

const MATE_SCORE_BASE: i32 = 10000;
// const MATE_DISTANCE_PENALTY: i32 = 10;

// TODO: Support mac and linux engine

#[derive(Debug, Clone)]
pub struct MoveWithScore {
	pub move_str: String,
	pub score: i32,
}

pub struct EngineWrapper {
	engine: ruci::Engine<BufReader<ChildStdout>, ChildStdin>,
}

impl EngineWrapper {
	pub fn new() -> Result<Self> {
		tracing::debug!("Engine binary found at {}", ENGINE_PATH);
		tracing::info!("Starting engine...");

		let mut process = Command::new(ENGINE_PATH)
			.arg(format!("--weights={}", DEFAULT_WEIGHTS_PATH))
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
			name: "MultiPV".into(),
			value: Some(CONFIG.engine.multi_pv.to_string().into()),
		})?;

		engine.send(ruci::SetOption {
			name: "Threads".into(),
			value: Some(CONFIG.engine.threads.to_string().into()),
		})?;

		engine.send(ruci::SetOption {
			name: "Hash".into(),
			value: Some(CONFIG.engine.hash.to_string().into()),
		})?;

		tracing::info!("Engine started successfully with config: {:?}", CONFIG);

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
				nodes: Some(CONFIG.engine.nodes),
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
					ruci::Score::Centipawns(cp) => cp as i32,
					ruci::Score::MateIn(moves) => {
						let sign = if moves > 0 { 1 } else { -1 };
						sign * (MATE_SCORE_BASE - moves.abs() as i32)
					}
				};

				match best_moves
					.iter_mut()
					.find(|m| m.move_str == first_move.to_string())
				{
					Some(existing) if score > existing.score => {
						existing.score = score;
					}
					None => {
						best_moves.push(MoveWithScore {
							move_str: first_move.to_string(),
							score,
						});
					}
					_ => {}
				}
			},
		)?;

		Ok(best_moves)
	}
}
