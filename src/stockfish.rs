use std::path::Path;

use anyhow::Result;

use crate::error::AnalysisError;

const MODELS_DIR: &str = "models";
const STOCKFISH_PATH: &str = "models/stockfish.exe";

const STOCKFISH_DOWNLOAD_URL: &str = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-avx2.zip";

const STOCKFISH_DEPTH: u32 = 12;
const STOCKFISH_MULTI_PV: usize = 3;

const MATE_SCORE_BASE: i32 = 10000;
const MATE_DISTANCE_PENALTY: i32 = 10;

// TODO: Check if fen is checkmate
// TODO: When downloading put the version of stockfish in the name for updating
// TODO: Support mac and linux stockfish

#[derive(Debug, Clone)]
pub struct MoveWithScore {
	pub move_str: String,
	pub score: i32,
}

pub struct StockfishWrapper {
	engine: stockfish::Stockfish,
}

impl StockfishWrapper {
	pub fn new() -> Result<Self> {
		Self::ensure_downloaded()?;

		tracing::debug!("Stockfish binary found at {}", STOCKFISH_PATH);
		tracing::info!("Starting Stockfish engine...");

		let mut engine = stockfish::Stockfish::new(STOCKFISH_PATH)
			.map_err(|e| AnalysisError::EngineStartFailed(e.to_string()))?;

		engine
			.setup_for_new_game()
			.map_err(|e| AnalysisError::EngineInitFailed(e.to_string()))?;

		engine.set_depth(STOCKFISH_DEPTH);
		engine
			.set_option("MultiPV", &STOCKFISH_MULTI_PV.to_string())
			.map_err(|_e| AnalysisError::EngineOptionFailed {
				option: "MultiPV".to_string(),
				value: STOCKFISH_MULTI_PV.to_string(),
			})?;

		tracing::info!("Stockfish engine started successfully");

		Ok(Self { engine })
	}

	fn ensure_downloaded() -> Result<()> {
		if Path::new(STOCKFISH_PATH).exists() {
			tracing::debug!("Stockfish binary found at {}", STOCKFISH_PATH);
			return Ok(());
		}

		tracing::info!("Stockfish not found, downloading...");

		std::fs::create_dir_all(MODELS_DIR).map_err(|e| {
			AnalysisError::IoError(format!("Failed to create models directory: {}", e))
		})?;

		Self::download()?;

		Ok(())
	}

	fn download() -> Result<()> {
		tracing::info!("Downloading Stockfish from {}", STOCKFISH_DOWNLOAD_URL);

		let response = reqwest::blocking::get(STOCKFISH_DOWNLOAD_URL).map_err(|e| {
			AnalysisError::DownloadFailed {
				url: STOCKFISH_DOWNLOAD_URL.to_string(),
				reason: e.to_string(),
			}
		})?;

		if !response.status().is_success() {
			return Err(AnalysisError::DownloadFailed {
				url: STOCKFISH_DOWNLOAD_URL.to_string(),
				reason: format!("HTTP {}", response.status()),
			}
			.into());
		}

		let bytes = response
			.bytes()
			.map_err(|e| AnalysisError::NetworkError(e.to_string()))?;

		tracing::info!("Downloaded {} bytes, extracting...", bytes.len());

		let cursor = std::io::Cursor::new(bytes);
		let mut archive = zip::ZipArchive::new(cursor)
			.map_err(|e| AnalysisError::ExtractionFailed(format!("Failed to open ZIP: {}", e)))?;

		for i in 0..archive.len() {
			let mut file = archive.by_index(i).map_err(|e| {
				AnalysisError::ExtractionFailed(format!("Failed to read ZIP entry: {}", e))
			})?;
			let file_name = file.name().to_string();

			if file_name.to_lowercase().contains("stockfish") && file_name.ends_with(".exe") {
				tracing::info!("Extracting {} to {}", file_name, STOCKFISH_PATH);

				let mut buffer = Vec::new();
				std::io::copy(&mut file, &mut buffer).map_err(|e| {
					AnalysisError::ExtractionFailed(format!("Failed to extract binary: {}", e))
				})?;

				std::fs::write(STOCKFISH_PATH, buffer).map_err(|e| {
					AnalysisError::IoError(format!("Failed to write Stockfish binary: {}", e))
				})?;

				tracing::info!("Stockfish downloaded and extracted successfully");
				return Ok(());
			}
		}

		Err(AnalysisError::ExecutableNotFoundInArchive.into())
	}

	pub fn set_position(&mut self, fen: &str) -> Result<()> {
		self.engine.set_fen_position(fen).map_err(|e| {
			AnalysisError::InvalidPosition {
				fen: fen.to_string(),
				reason: e.to_string(),
			}
			.into()
		})
	}

	pub fn get_best_moves(&mut self) -> Result<Vec<MoveWithScore>> {
		tracing::trace!("Requesting best moves from Stockfish");

		let output = self
			.engine
			.go_multipv()
			.map_err(|e| AnalysisError::AnalysisFailed(e.to_string()))?;

		let mut best_moves = std::collections::HashMap::new();

		for pv_line in output.pv_lines() {
			if let Some(move_str) = pv_line.first_move() {
				let eval = pv_line.eval();
				let score = match eval.eval_type() {
					stockfish::EvalType::Centipawn => eval.value(),
					stockfish::EvalType::Mate => {
						if eval.value() > 0 {
							MATE_SCORE_BASE - eval.value().abs() * MATE_DISTANCE_PENALTY
						} else {
							-MATE_SCORE_BASE + eval.value().abs() * MATE_DISTANCE_PENALTY
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

		moves.truncate(STOCKFISH_MULTI_PV);

		Ok(moves)
	}
}
