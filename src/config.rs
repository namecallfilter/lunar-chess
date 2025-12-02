use std::collections::HashMap;

use anyhow::Result;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

use crate::model::Confidence;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
	pub engine: EngineConfig,
	pub profiles: HashMap<String, EngineProfile>,
	pub detection: DetectionConfig,
	pub debugging: DebuggingConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DetectionConfig {
	pub path: String,
	pub piece_confidence_threshold: Confidence,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EngineConfig {
	pub path: String,
	pub args: Vec<String>,
	pub book: Option<String>,
	pub profile: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EngineProfile {
	#[serde(default)]
	pub uci: HashMap<String, String>,
	#[serde(default)]
	pub go: Option<GoConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GoConfig {
	pub search_moves: Option<Vec<String>>,
	pub ponder: Option<bool>,
	pub wtime: Option<usize>,
	pub btime: Option<usize>,
	pub winc: Option<usize>,
	pub binc: Option<usize>,
	pub movestogo: Option<usize>,
	pub depth: Option<usize>,
	pub nodes: Option<usize>,
	pub mate: Option<usize>,
	pub movetime: Option<usize>,
	pub infinite: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DebuggingConfig {
	pub level: String,
	pub stream_proof: bool,
	pub show_grid: bool,
	pub show_piece_labels: bool,
}

pub static CONFIG: Lazy<Config> =
	Lazy::new(|| Config::load().expect("Failed to load configuration"));

pub static PROFILE: Lazy<EngineProfile> = Lazy::new(|| {
	CONFIG
		.profiles
		.get(&CONFIG.engine.profile)
		.cloned()
		.expect("Profile not found in config")
});

impl Config {
	pub fn load() -> Result<Self> {
		let settings = config::Config::builder()
			.add_source(config::File::with_name("config").required(true))
			.add_source(config::Environment::with_prefix(env!("CARGO_CRATE_NAME")))
			.build()?;

		settings.try_deserialize().map_err(Into::into)
	}
}
