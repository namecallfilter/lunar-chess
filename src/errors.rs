use thiserror::Error;

#[derive(Error, Debug)]
pub enum CaptureError {
	#[error("Screen capture failed: {0}")]
	CaptureFailed(String),
}

#[derive(Error, Debug)]
pub enum AnalysisError {
	#[error("Failed to start engine: {0}")]
	EngineStartFailed(String),

	#[error("Invalid FEN position '{fen}': {reason}")]
	InvalidPosition { fen: String, reason: String },
}

#[derive(Error, Debug)]
pub enum BoardToFenError {
	#[error("Missing king: {color} king not found on board")]
	MissingKing { color: &'static str },

	#[error("Board orientation is unknown, cannot generate FEN")]
	UnknownOrientation,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fen(String);

impl Fen {
	#[inline]
	pub fn as_str(&self) -> &str {
		&self.0
	}

	pub(crate) fn from_validated(fen: String) -> Self {
		Self(fen)
	}
}

impl std::fmt::Display for Fen {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.0)
	}
}

impl AsRef<str> for Fen {
	fn as_ref(&self) -> &str {
		&self.0
	}
}
