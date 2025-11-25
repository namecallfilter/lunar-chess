# Lunar Chess

A screen overlay that detects chessboards, recognizes pieces using ONNX, and displays best moves from a chess engine.

## Setup

1. Copy `config.example.toml` to `config.toml`
2. Configure the required paths:
   - `engine.path` - Path to your UCI chess engine executable
   - `detection.path` - Path to the ONNX model for piece detection

## Configuration

```toml
[engine]
path = "path/to/engine.exe"
args = []
profile = "default"

[profiles.default]
threads = 8
hash = 8192
multi_pv = 6
depth = 5

[detection]
path = "path/to/model.onnx"
piece_confidence_threshold = 0.75

[debugging]
level = "info"
stream_proof = true
show_grid = false
show_piece_labels = false
```

## Usage

Run the application and it will automatically detect any chessboard on screen and overlay the best moves.
