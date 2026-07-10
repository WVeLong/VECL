from pathlib import Path
import runpy

TARGET = Path(__file__).resolve().parents[1] / "downstream/retrieval/retrieval_post_process.py"
runpy.run_path(str(TARGET), run_name="__main__")
