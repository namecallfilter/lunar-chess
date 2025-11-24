use thiserror::Error;

#[derive(Error, Debug)]
pub enum CaptureError {
	#[error("Screen capture failed: {0}")]
	CaptureFailed(String),
}

#[derive(Error, Debug)]
pub enum AnalysisError {
	#[error("Failed to start Stockfish engine: {0}")]
	EngineStartFailed(String),

	#[error("Invalid FEN position '{fen}': {reason}")]
	InvalidPosition { fen: String, reason: String },
}
