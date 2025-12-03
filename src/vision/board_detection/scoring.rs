use crate::{model::detected::Rect, vision::board_detection::edges::EdgeMap};

pub const GRID_LINES: usize = 9;

pub fn calculate_grid_score(rect: &Rect, edge_map: &EdgeMap) -> f32 {
	let mut hits: usize = 0;
	let mut total: usize = 0;
	let cols = (GRID_LINES - 1) as f32;
	let rows = (GRID_LINES - 1) as f32;

	let x0 = rect.x();
	let y0 = rect.y();
	let w = rect.width();
	let h = rect.height();

	if w <= 0.0 || h <= 0.0 {
		return 0.0;
	}

	let map_w = edge_map.width();
	let map_h = edge_map.height();

	let max_samples = 400usize;
	let vert_len = (h as usize).max(1);
	let hor_len = (w as usize).max(1);

	let v_stride = if vert_len > max_samples {
		((vert_len as f32 / max_samples as f32).ceil() as usize).max(1)
	} else {
		1
	};

	let h_stride = if hor_len > max_samples {
		((hor_len as f32 / max_samples as f32).ceil() as usize).max(1)
	} else {
		1
	};

	for i in 0..GRID_LINES {
		let fx = x0 + (i as f32) * (w / cols);
		let xi = fx.round() as isize;
		let mut j = 0usize;
		while j < vert_len {
			let yi = (y0.round() as isize) + (j as isize);
			if xi >= 0 && yi >= 0 {
				let xu = xi as usize;
				let yu = yi as usize;
				if xu < map_w && yu < map_h {
					total += 1;

					if edge_map.is_edge(xu, yu) {
						hits += 1;
					}
				}
			}

			j = j.saturating_add(v_stride);
		}
	}

	for i in 0..GRID_LINES {
		let fy = y0 + (i as f32) * (h / rows);
		let yi = fy.round() as isize;
		let mut j = 0usize;
		while j < hor_len {
			let xi = (x0.round() as isize) + (j as isize);
			if xi >= 0 && yi >= 0 {
				let xu = xi as usize;
				let yu = yi as usize;
				if xu < map_w && yu < map_h {
					total += 1;

					if edge_map.is_edge(xu, yu) {
						hits += 1;
					}
				}
			}

			j = j.saturating_add(h_stride);
		}
	}

	if total == 0 {
		0.0
	} else {
		(hits as f32) / (total as f32)
	}
}
