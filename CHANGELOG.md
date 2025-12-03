# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.4](https://github.com/namecallfilter/lunar-chess/compare/v0.2.3...v0.2.4) - 2025-12-03

### Added

- *(engine)* only show mate moves if you have a mate-in <x> ([#27](https://github.com/namecallfilter/lunar-chess/pull/27))

### Fixed

- *(board_detection)* Incorrectly inlcuding eval bar ([#29](https://github.com/namecallfilter/lunar-chess/pull/29))

### Other

- *(board_detection)* change into a more modular structure ([#30](https://github.com/namecallfilter/lunar-chess/pull/30))

## [0.2.3](https://github.com/namecallfilter/lunar-chess/compare/v0.2.2...v0.2.3) - 2025-12-02

### Other

- performance and style enhancments  ([#26](https://github.com/namecallfilter/lunar-chess/pull/26))
- *(analysis)* fix speed of engine and board analysis ([#25](https://github.com/namecallfilter/lunar-chess/pull/25))
- massive cleanup ([#24](https://github.com/namecallfilter/lunar-chess/pull/24))
- fix sematic check running on push ([#22](https://github.com/namecallfilter/lunar-chess/pull/22))

## [0.2.2](https://github.com/namecallfilter/lunar-chess/compare/v0.2.1...v0.2.2) - 2025-11-26

### Other

- release-plz will now release with the exe ([#20](https://github.com/namecallfilter/lunar-chess/pull/20))

## [0.2.1](https://github.com/namecallfilter/lunar-chess/compare/v0.2.0...v0.2.1) - 2025-11-26

### Other

- release v0.2.0 ([#16](https://github.com/namecallfilter/lunar-chess/pull/16))

## [0.2.0](https://github.com/namecallfilter/lunar-chess/releases/tag/v0.2.0) - 2025-11-26

### Added

- add edge refinement for better board detection ([#8](https://github.com/namecallfilter/lunar-chess/pull/8))
- lint and release workflows
- supported platforms
- graceful shutdown
- config example
- profiles
- optimize vision
- stockfish added

### Fixed

- release-plz ([#15](https://github.com/namecallfilter/lunar-chess/pull/15))
- trigger release with file change ([#13](https://github.com/namecallfilter/lunar-chess/pull/13))
- remove push run ([#3](https://github.com/namecallfilter/lunar-chess/pull/3))
- use PAT
- use PAT
- opasity based on scoring not step
- truncate multipv & PROFILE static
- little more accurate
- double buffer micro optimization
- parking lot
- loads label font once
- parking lot
- config
- player orientation
- chessboard in noisy area
- made grid detection more accurate
- board orientation

### Other

- add mit license and Cargo description ([#18](https://github.com/namecallfilter/lunar-chess/pull/18))
- run the workflow on windows to fix the error ([#17](https://github.com/namecallfilter/lunar-chess/pull/17))
- manual bump to v0.2.0 to trigger release ([#14](https://github.com/namecallfilter/lunar-chess/pull/14))
- resolve publish conflict in release-plz config ([#11](https://github.com/namecallfilter/lunar-chess/pull/11))
- fix release-plz toml syntax ([#10](https://github.com/namecallfilter/lunar-chess/pull/10))
- force release-plz to track lunar-chess ([#9](https://github.com/namecallfilter/lunar-chess/pull/9))
- finalize release workflow with draft support ([#7](https://github.com/namecallfilter/lunar-chess/pull/7))
- release v0.1.0 ([#5](https://github.com/namecallfilter/lunar-chess/pull/5))
- disable crates.io publishing ([#6](https://github.com/namecallfilter/lunar-chess/pull/6))
- ci/update workflows ([#4](https://github.com/namecallfilter/lunar-chess/pull/4))
- cleanup ([#2](https://github.com/namecallfilter/lunar-chess/pull/2))
- apply type-driven design patterns
- add strip
- add readme
- profile fn --> PROFILE static
- fmt
- Delete config.toml
- Delete models directory
- update
- update .gitignore
- refactor logging
- file structure and screenshot
- cba
- *(vision)* rename
- shouldnt be commitewd
- *(board_detection)* Made it use an algorithm instead of an ML.
- cleanup
- cleanup
- multipv with ruci and training folder
- refactor
- cleanup
- init working
- added fen
- clean
- cleanup
- initial
- temp
- ON EVERYTHING IM GOING TO CRASHOUT
- Merge branch 'main' of https://github.com/kaorlol/lunar-chess
- cleanup
- opencv added
- ready for opencv in dynboard
- Add TODO comments
- first commit

## [0.1.0](https://github.com/namecallfilter/lunar-chess/releases/tag/v0.1.0) - 2025-11-26

### Other

- ci/update workflows ([#4](https://github.com/namecallfilter/lunar-chess/pull/4))
