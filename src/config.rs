use std::collections::HashMap;

use anyhow::{Context, Result};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

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
	pub piece_confidence_threshold: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EngineConfig {
	pub path: String,
	pub args: Vec<String>,
	pub profile: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EngineProfile {
	pub threads: usize,
	pub hash: usize,
	pub multi_pv: usize,
	pub depth: usize,
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

impl Config {
	pub fn load() -> Result<Self> {
		let settings = config::Config::builder()
			.add_source(config::File::with_name("config").required(true))
			.add_source(config::Environment::with_prefix(env!("CARGO_CRATE_NAME")))
			.build()?;

		settings.try_deserialize().map_err(Into::into)
	}

	pub fn active_profile(&self) -> Result<&EngineProfile> {
		self.profiles.get(&self.engine.profile).context(format!(
			"Profile '{}' not found in config",
			self.engine.profile
		))
	}
}
