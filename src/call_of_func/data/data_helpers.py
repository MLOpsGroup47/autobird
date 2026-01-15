from pathlib import Path

import typer
from call_of_func.dataclasses.pathing import PathConfig

root = Path(__file__).resolve().parents[2]  # project root

# cfg = PathConfig(
#    root=root
# )


def rn_dir(root: Path = typer.Argument("data/voice_of_birds", exists=True)) -> None:
    """Rename directories to replace spaces with underscores.

    Arg:
        root: dir to rename
    """
    # rename dirs in root
    for p in sorted(root.iterdir()):
        if p.is_dir():
            new_name = p.name.replace(" ", "_")
            if new_name != p.name:
                p.rename(p.with_name(new_name))
            print(f"{p.name} -> {new_name}")


def rn_mp3(cfg, exists=True) -> None:
    """Rename audio files to replace spaces with underscores.

    Arg:
        root: dir to rename
    """
    # rename files in root
    for f in cfg.root.rglob("*"):
        if f.is_file() and f.suffix.lower() in cfg.audio_exts:
            new_name = f.name.replace(" ", "_")
            if new_name != f.name:
                f.rename(f.with_name(new_name))
                print(f"{f.name} -> {new_name}")
