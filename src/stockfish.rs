use std::path::Path;

use anyhow::{Context, Result, bail};

const STOCKFISH_URL: &str = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-avx2.zip";
const STOCKFISH_PATH: &str = "models/stockfish.exe";
const MODELS_DIR: &str = "models";
const MULTI_PV: usize = 3;

#[derive(Debug, Clone)]
pub struct MoveWithScore {
	pub move_str: String,
	pub score: i32,
}

pub struct StockfishWrapper {
	engine: stockfish::Stockfish,
}

// TODO: Check if fen is checkmate
// TODO: When downloading put the version of stockfish in the name for updating
// TODO: Support mac and linux stockfish

impl StockfishWrapper {
	pub fn new() -> Result<Self> {
		Self::ensure_downloaded()?;

		tracing::debug!("Stockfish binary found at {}", STOCKFISH_PATH);
		tracing::info!("Starting Stockfish engine...");

		let mut engine = stockfish::Stockfish::new(STOCKFISH_PATH)
			.context("Failed to start Stockfish engine")?;

		engine
			.setup_for_new_game()
			.context("Failed to setup Stockfish for new game")?;

		// Configure Stockfish
		engine.set_depth(12); // Good balance of speed and accuracy
		engine.set_option("MultiPV", &MULTI_PV.to_string())?; // Get top N moves

		tracing::info!("Stockfish engine started successfully");

		Ok(Self { engine })
	}

	fn ensure_downloaded() -> Result<()> {
		if Path::new(STOCKFISH_PATH).exists() {
			tracing::debug!("Stockfish binary found at {}", STOCKFISH_PATH);
			return Ok(());
		}

		tracing::info!("Stockfish not found, downloading...");

		std::fs::create_dir_all(MODELS_DIR).context("Failed to create models directory")?;

		Self::download()?;

		Ok(())
	}

	fn download() -> Result<()> {
		tracing::info!("Downloading Stockfish from {}", STOCKFISH_URL);

		let response =
			reqwest::blocking::get(STOCKFISH_URL).context("Failed to download Stockfish")?;

		if !response.status().is_success() {
			bail!("Failed to download Stockfish: HTTP {}", response.status());
		}

		let bytes = response
			.bytes()
			.context("Failed to read Stockfish download")?;

		tracing::info!("Downloaded {} bytes, extracting...", bytes.len());

		let cursor = std::io::Cursor::new(bytes);
		let mut archive = zip::ZipArchive::new(cursor).context("Failed to open ZIP archive")?;

		for i in 0..archive.len() {
			let mut file = archive.by_index(i).context("Failed to read ZIP entry")?;
			let file_name = file.name().to_string();

			if file_name.to_lowercase().contains("stockfish") && file_name.ends_with(".exe") {
				tracing::info!("Extracting {} to {}", file_name, STOCKFISH_PATH);

				let mut buffer = Vec::new();
				std::io::copy(&mut file, &mut buffer)
					.context("Failed to extract Stockfish binary")?;

				std::fs::write(STOCKFISH_PATH, buffer)
					.context("Failed to write Stockfish binary")?;

				tracing::info!("Stockfish downloaded and extracted successfully");
				return Ok(());
			}
		}

		bail!("Stockfish executable not found in downloaded archive");
	}

	pub fn set_position(&mut self, fen: &str) -> Result<()> {
		self.engine
			.set_fen_position(fen)
			.context("Failed to set FEN position")?;
		Ok(())
	}

	pub fn get_best_moves(&mut self) -> Result<Vec<MoveWithScore>> {
		tracing::trace!("Requesting best moves from Stockfish");

		let output = self
			.engine
			.go_multipv()
			.context("Failed to get engine output")?;

		let mut best_moves = std::collections::HashMap::new();

		for pv_line in output.pv_lines() {
			if let Some(move_str) = pv_line.first_move() {
				let eval = pv_line.eval();
				let score = match eval.eval_type() {
					stockfish::EvalType::Centipawn => eval.value(),
					stockfish::EvalType::Mate => {
						if eval.value() > 0 {
							10000 - eval.value().abs() * 10
						} else {
							-10000 + eval.value().abs() * 10
						}
					}
				};

				best_moves
					.entry(move_str.clone())
					.and_modify(|existing_score| {
						if score > *existing_score {
							*existing_score = score;
						}
					})
					.or_insert(score);
			}
		}

		let mut moves: Vec<MoveWithScore> = best_moves
			.into_iter()
			.map(|(move_str, score)| MoveWithScore { move_str, score })
			.collect();

		moves.sort_by(|a, b| b.score.cmp(&a.score));

		moves.truncate(MULTI_PV);

		Ok(moves)
	}
}
