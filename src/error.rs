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

#[allow(dead_code)]
#[derive(Error, Debug)]
pub enum DetectionError {
	#[error("Failed to load model from '{path}': {reason}")]
	ModelLoadFailed { path: String, reason: String },

	#[error("ONNX inference failed: {0}")]
	InferenceFailed(String),

	#[error("Image preprocessing failed: {0}")]
	PreprocessingFailed(String),

	#[error("Invalid YOLO output shape: expected {expected}, got {got}")]
	InvalidOutputShape { expected: String, got: String },

	#[error("Image resize failed: {0}")]
	ResizeFailed(String),

	#[error("Tensor conversion failed: {0}")]
	TensorConversionFailed(String),

	#[error("CUDA initialization failed: {0}")]
	CudaInitFailed(String),
}

#[derive(Error, Debug)]
pub enum AnalysisError {
	#[error("Failed to download Stockfish from '{url}': {reason}")]
	DownloadFailed { url: String, reason: String },

	#[error("Failed to extract Stockfish: {0}")]
	ExtractionFailed(String),

	#[error("Stockfish executable not found in downloaded archive")]
	ExecutableNotFoundInArchive,

	#[error("Failed to start Stockfish engine: {0}")]
	EngineStartFailed(String),

	#[error("Failed to initialize engine: {0}")]
	EngineInitFailed(String),

	#[error("Failed to set engine option '{option}' to '{value}'")]
	EngineOptionFailed { option: String, value: String },

	#[error("Invalid FEN position '{fen}': {reason}")]
	InvalidPosition { fen: String, reason: String },

	#[error("Engine analysis failed: {0}")]
	AnalysisFailed(String),

	#[error("Network error: {0}")]
	NetworkError(String),

	#[error("I/O error: {0}")]
	IoError(String),
}

#[allow(dead_code)]
#[derive(Error, Debug)]
pub enum RenderError {
	#[error("Failed to create window: {0}")]
	WindowCreationFailed(String),

	#[error("Failed to create graphics context: {0}")]
	ContextCreationFailed(String),

	#[error("Failed to create draw target: {0}")]
	DrawTargetCreationFailed(String),

	#[error("Failed to load font: {0}")]
	FontLoadFailed(String),

	#[error("Invalid overlay dimensions: {width}x{height}")]
	InvalidDimensions { width: u32, height: u32 },
}
