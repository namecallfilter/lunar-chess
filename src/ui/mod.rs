pub mod dimensions;
pub mod draw_board;
pub mod overlay;
pub mod types;

use std::sync::Arc;

pub use dimensions::{CellSize, ScreenDimensions};
pub use overlay::{UserEvent, start_overlay};
use parking_lot::Mutex;
pub use types::{ArrowStyle, Color, Point2D, Vec2};

use crate::model::detected::BoardState;

pub type SharedBoardState = Arc<Mutex<Option<BoardState>>>;
