pub mod draw_board;
pub mod overlay;

use std::sync::{Arc, Mutex};

pub use overlay::{BestMove, BoardBounds, UserEvent, start_overlay};

use crate::model::detected::{DetectedBoard, DetectedPiece};

pub type SharedBoardState = Arc<Mutex<Option<(DetectedBoard, Vec<DetectedPiece>)>>>;
