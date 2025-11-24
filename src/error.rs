use thiserror::Error;

#[derive(Error, Debug)]
pub enum CaptureError {
	#[error("No monitors detected on the system")]
	NoMonitorsFound,

	#[error("Could not identify the primary monitor")]
	NoPrimaryMonitor,

	#[error("Screen capture failed: {0}")]
	CaptureFailed(String),

	#[error("Invalid monitor configuration: {0}")]
	InvalidMonitor(String),
}

#[derive(Error, Debug)]
pub enum AnalysisError {
	#[error("Failed to start Stockfish engine: {0}")]
	EngineStartFailed(String),

	#[error("Invalid FEN position '{fen}': {reason}")]
	InvalidPosition { fen: String, reason: String },
}
