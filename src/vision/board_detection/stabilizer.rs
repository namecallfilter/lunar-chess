use crate::model::detected::{DetectedBoard, Rect};

pub struct BoardStabilizer {
	smoothed_rect: Option<Rect>,
	alpha: f32,
}

impl BoardStabilizer {
	pub fn new(smoothing_factor: f32) -> Self {
		Self {
			smoothed_rect: None,
			alpha: smoothing_factor.clamp(0.0, 1.0),
		}
	}

	pub fn update(&mut self, new_detection: Option<DetectedBoard>) -> Option<DetectedBoard> {
		match (self.smoothed_rect, new_detection) {
			(Some(prev), Some(curr)) => {
				if (prev.x() - curr.rect.x()).abs() > 50.0
					|| (prev.width() - curr.rect.width()).abs() > 50.0
				{
					self.smoothed_rect = Some(curr.rect);
				} else {
					let new_x = prev.x() + (curr.rect.x() - prev.x()) * self.alpha;
					let new_y = prev.y() + (curr.rect.y() - prev.y()) * self.alpha;
					let new_w = prev.width() + (curr.rect.width() - prev.width()) * self.alpha;
					let new_h = prev.height() + (curr.rect.height() - prev.height()) * self.alpha;

					self.smoothed_rect = Some(Rect::new(new_x, new_y, new_w, new_h));
				}
			}
			(None, Some(curr)) => {
				self.smoothed_rect = Some(curr.rect);
			}
			(_, None) => {
				self.smoothed_rect = None;
			}
		}

		self.smoothed_rect.map(DetectedBoard::new)
	}
}
