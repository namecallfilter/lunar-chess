use std::{borrow::Cow, path::Path, process::Stdio};

use anyhow::{Context, Result, bail};
use ruci::{Engine, Go, Position};
use shakmaty::fen::Fen;
use tokio::{
	io::{AsyncBufReadExt as _, BufReader},
	process::{ChildStdin, ChildStdout, Command},
};

pub struct Stockfish {
	engine: Engine<BufReader<ChildStdout>, ChildStdin>,
	move_time: Option<usize>,
}

impl Stockfish {
	pub async fn new(path: &str) -> Result<Self> {
		if !Path::new(path).exists() {
			bail!("Stockfish path does not exist: {}", path);
		}

		let mut process = Command::new(path)
			.stdout(Stdio::piped())
			.stdin(Stdio::piped())
			.spawn()?;

		let mut engine = Engine::from_process_async(&mut process, true)?;
		engine.engine.read_line(&mut String::new()).await?;

		Ok(Self {
			engine,
			move_time: Some(100),
		})
	}

	pub async fn set_position(&mut self, fen: &str) -> Result<()> {
		self.engine
			.send_async(Position::Fen {
				fen: Cow::Owned(Fen::from_ascii(fen.as_bytes())?),
				moves: Cow::Borrowed(&[]),
			})
			.await?;
		Ok(())
	}

	pub async fn get_best_move(&mut self) -> Result<String> {
		let go = Go {
			move_time: self.move_time,
			..Default::default()
		};

		let best_move = self
			.engine
			.go_async(&go, async |_info| async {}.await)
			.await?;

		let normal = best_move
			.take_normal()
			.context("Failed to get normal move")?;

		Ok(normal.r#move.to_string())
	}
}
