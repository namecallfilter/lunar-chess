pub mod draw_board;
pub mod overlay;

use std::sync::{Arc, Mutex};
use crate::model::detected::{DetectedBoard, DetectedPiece};

pub use overlay::{BestMove, BoardBounds, UserEvent, start_overlay};

pub type SharedBoardState = Arc<Mutex<Option<(DetectedBoard, Vec<DetectedPiece>)>>>;
