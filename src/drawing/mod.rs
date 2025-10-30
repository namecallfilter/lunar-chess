mod board;
mod overlay;

use std::sync::{Arc, Mutex};

pub use board::{DetectedBoard, DetectedPiece};
pub use overlay::{BestMove, BoardBounds, UserEvent, start_overlay};

pub type SharedBoardState = Arc<Mutex<Option<(DetectedBoard, Vec<DetectedPiece>)>>>;
