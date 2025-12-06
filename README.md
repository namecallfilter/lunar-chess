# Lunar Chess

A screen overlay that detects chessboards, recognizes pieces using ONNX, and displays best moves from a chess engine.

## Supported Platforms

| Platform | Status |
|----------|--------|
| Windows  | Supported |
| macOS    | Supported |
| Linux    | Cannot support due to X11 and Wayland limitations |

## Installation

### Option 1: Download Binary (Recommended)
Download the latest executable for Windows or macOS from the **[Releases Page](https://github.com/namecallfilter/lunar-chess/releases/latest)**.

### Option 2: Install via Cargo
If you have Rust installed, you can install directly from crates.io:
```bash
cargo install lunar-chess
```

## Setup

After installing, you need to set up the configuration and model:

1.  **Download the Model:** Get the ONNX model file from [Hugging Face](https://huggingface.co/KaoruLOL/lunar-chess-piece-detector/blob/main/piece.onnx).
2.  **Create Config:** Create a `config.toml` file in the same folder as the executable (copy the example below).
3.  **Configure Paths:** Open `config.toml` and set:
      - `engine.path` - Path to your UCI chess engine executable (e.g., Stockfish).
      - `detection.path` - Path to the `piece.onnx` file you downloaded.

## Configuration (`config.toml`)

```toml
[engine]
path = "" # Path to the exe of the engine
args = [] # Any args you give the engine like '--weights='
book = "" # Optional: Path to a .bin polyglot opening book
profile = "default" # The profile to use from [profiles.<name>]

# Profiles to change the UCI options and go
[profiles.default]
# UCI options are passed directly to the engine via "setoption"
# See https://backscattering.de/chess/uci/#engine-option for standard options
uci.Threads = "8"
uci.Hash = "8192"
uci.MultiPV = "6"

# Go commands control the search parameters
# See https://backscattering.de/chess/uci/#gui-go for available commands
go.movetime = 10000

[detection]
# To use my model see https://huggingface.co/KaoruLOL/lunar-chess-piece-detector/blob/main/piece.onnx
path = "" # Path to the ONNX file that detect the pieces
piece_confidence_threshold = 0.75 # Threshold to show the pieces based on its confidence

[debugging]
level = "info" # Level to show the logs: info, debug, warn, error, trace
show_grid = false # Shows the detected grid
show_piece_labels = false # Shows which pieces are labeled what
```

## Usage

Run the application and it will automatically detect any chessboard on screen and overlay the best moves.
