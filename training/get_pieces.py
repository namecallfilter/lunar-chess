import asyncio
import os
import shutil
import stat
from pathlib import Path

import aiohttp
import cairosvg
import git
from tqdm.asyncio import tqdm as tqdm_async

# https://www.reddit.com/r/DataHoarder/comments/1lbhxxv/chesscom_svg_assets/
# https://lichess.org/forum/general-chess-discussion/is-there-somewhere-i-can-download-the-piece-sets-that-lichess-uses

baseUrl = "https://images.chesscomfiles.com/chess-themes/pieces"
outputDir = Path("./assets/pieces")

colors = ["w", "b"]
pieces = ["k", "q", "r", "b", "n", "p"]

chesscom_piece_themes = [
	"neo",
	"game_room",
	"wood",
	"glass",
	"gothic",
	"classic",
	"metal",
	"bases",
	"neo_wood",
	"icy_sea",
	"club",
	"ocean",
	"newspaper",
	"blindfold",
	"space",
	"cases",
	"condal",
	"3d_chesskid",
	"8_bit",
	"marble",
	"book",
	"alpha",
	"bubblegum",
	"dash",
	"graffiti",
	"light",
	"lolz",
	"luca",
	"maya",
	"modern",
	"nature",
	"neon",
	"sky",
	"tigers",
	"tournament",
	"vintage",
	"3d_wood",
	"3d_staunton",
	"3d_plastic",
	"real_3d",
]


async def download_chesscom_piece(
	session: aiohttp.ClientSession, set_name: str, color: str, piece: str
) -> bool:
	url = f"{baseUrl}/{set_name}/150/{color}{piece}.png"
	save_path = outputDir / set_name / f"{color}{piece}.png"

	try:
		async with session.get(url) as response:
			if response.status == 200:
				save_path.parent.mkdir(parents=True, exist_ok=True)
				content = await response.read()
				save_path.write_bytes(content)
				return True
			else:
				return False
	except Exception:
		return False


async def download_chesscom_pieces():
	print("Starting Chess.com downloads...")

	tasks = []
	connector = aiohttp.TCPConnector(limit=100)
	timeout = aiohttp.ClientTimeout(total=30)

	async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
		for set_name in chesscom_piece_themes:
			for color in colors:
				for piece in pieces:
					tasks.append(
						download_chesscom_piece(session, set_name, color, piece)
					)

		results = []
		for coro in tqdm_async.as_completed(tasks, desc="Downloading Chess.com pieces"):
			result = await coro
			results.append(result)

	success_count = sum(results)
	print(f"Chess.com downloads complete: {success_count}/{len(tasks)} successful")


def on_rm_error(func, path, exc_info):
	os.chmod(path, stat.S_IWRITE)
	func(path)


async def download_lichess_pieces():
	repo_url = "https://github.com/lichess-org/lila.git"
	repo_dir = Path("temp_lila_repo")
	target_subfolder = "public/piece"
	output_folder = outputDir

	print("Cloning Lichess repo (this may take a moment)...")
	if repo_dir.exists():
		shutil.rmtree(repo_dir, onerror=on_rm_error)

	git.Repo.clone_from(repo_url, str(repo_dir), depth=1)
	print("Clone complete. Converting SVGs to PNGs...")
	source_path = repo_dir / target_subfolder

	# Collect all SVG files - dynamically process all themes in the repo
	svg_files = []
	for root, dirs, files in os.walk(source_path):
		# Skip the root directory itself, only process subdirectories
		if Path(root) == source_path:
			continue

		for file in files:
			if file.lower().endswith(".svg"):
				rel_dir = os.path.relpath(root, source_path)
				output_dir_path = output_folder / rel_dir
				svg_path = Path(root) / file
				png_filename = os.path.splitext(file)[0] + ".png"
				png_path = output_dir_path / png_filename
				svg_files.append((svg_path, png_path, output_dir_path))

	from tqdm import tqdm

	print(f"Found {len(svg_files)} SVG files across all piece themes")

	failed_count = 0
	for svg_path, png_path, output_dir_path in tqdm(
		svg_files, desc="Converting Lichess SVGs"
	):
		try:
			output_dir_path.mkdir(parents=True, exist_ok=True)
			cairosvg.svg2png(
				url=str(svg_path),
				write_to=str(png_path),
				output_height=150,
				output_width=150,
			)
		except Exception:
			failed_count += 1

	if failed_count > 0:
		print(
			f"Note: {failed_count} SVG files failed to convert (likely invalid syntax)"
		)

	print("Lichess conversion complete. Cleaning up repo...")
	shutil.rmtree(repo_dir, onerror=on_rm_error)
	print("Cleanup complete.")


async def main():
	outputDir.mkdir(parents=True, exist_ok=True)

	await asyncio.gather(download_chesscom_pieces(), download_lichess_pieces())

	print(f"All assets downloaded to {outputDir}")


if __name__ == "__main__":
	asyncio.run(main())
