use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Confidence(f32);

impl Confidence {
	#[inline]
	pub fn new(value: f32) -> Self {
		Self(value.clamp(0.0, 1.0))
	}

	#[inline]
	pub const fn value(self) -> f32 {
		self.0
	}

	#[inline]
	pub fn as_percentage(self) -> f32 {
		self.0 * 100.0
	}
}

impl std::fmt::Display for Confidence {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{:.1}%", self.as_percentage())
	}
}

impl From<f32> for Confidence {
	fn from(value: f32) -> Self {
		Self::new(value)
	}
}

impl Serialize for Confidence {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: Serializer,
	{
		self.0.serialize(serializer)
	}
}

impl<'de> Deserialize<'de> for Confidence {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		let value = f32::deserialize(deserializer)?;
		Ok(Self::new(value))
	}
}
