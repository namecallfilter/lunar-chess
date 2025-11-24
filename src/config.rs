use anyhow::Result;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
	pub engine: EngineConfig,
	pub detection: DetectionConfig,
	pub debugging: DebuggingConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DetectionConfig {
	pub edge_threshold: f32,
	pub hough_threshold: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EngineConfig {
	pub threads: usize,
	pub hash: usize,
	pub multi_pv: usize,
	pub nodes: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DebuggingConfig {
	pub level: String,
	pub show_grid: bool,
	pub show_piece_labels: bool,
}

pub static CONFIG: Lazy<Config> =
	Lazy::new(|| Config::load().expect("Failed to load configuration"));

impl Config {
	pub fn load() -> Result<Self> {
		let settings = config::Config::builder()
			.add_source(config::File::with_name("config").required(true))
			.add_source(config::Environment::with_prefix("LUNAR_CHESS"))
			.build()?;

		settings.try_deserialize().map_err(Into::into)
	}
}
