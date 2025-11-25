pub mod draw_board;
pub mod overlay;

use std::sync::Arc;

pub use overlay::{BestMove, BoardBounds, UserEvent, start_overlay};
use parking_lot::Mutex;

use crate::model::detected::{DetectedBoard, DetectedPiece};

pub type SharedBoardState = Arc<Mutex<Option<(DetectedBoard, Vec<DetectedPiece>)>>>;
