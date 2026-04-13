import argparse
import os
import runpy
import sys
from typing import List, Optional

import modal
from modal import FilePatternMatcher

app = modal.App("parameter-golf-run-python-l4")

image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install_from_requirements("requirements.txt")
    # Mount only Python source code from the repo.
    .add_local_dir(".", "/workspace", ignore=~FilePatternMatcher("**/*.py"))
    # Mount training data/tokenizer from your local ./data folder.
    .add_local_dir("data", "/workspace/data")
)


@app.function(gpu="L4", image=image, timeout=60 * 60 * 12)
def run_python_file(
    *,
    script_path: str,
    script_args: List[str],
    data_path: str = "/workspace/data/datasets/fineweb10B_sp1024",
    tokenizer_path: str = "/workspace/data/tokenizers/fineweb_1024_bpe.model",
    run_id: str = "",
    iterations: Optional[int] = None,
    max_wallclock: int = 600,
) -> None:
    os.chdir("/workspace")
    os.environ["DATA_PATH"] = data_path
    os.environ["TOKENIZER_PATH"] = tokenizer_path
    os.environ["MAX_WALLCLOCK_SECONDS"] = str(max_wallclock)
    if run_id:
        os.environ["RUN_ID"] = run_id
    if iterations is not None:
        os.environ["ITERATIONS"] = str(iterations)

    if not os.path.exists("/workspace/data"):
        raise FileNotFoundError("Missing /workspace/data. Create a local ./data folder to mount.")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    argv = [script_path, *script_args]
    sys.argv = argv
    runpy.run_path(script_path, run_name="__main__")


@app.local_entrypoint()
def main(
    file: str = "train_gpt.py",
    data_path: str = "/workspace/data/datasets/fineweb10B_sp1024",
    tokenizer_path: str = "/workspace/data/tokenizers/fineweb_1024_bpe.model",
    run_id: str = "",
    iterations: Optional[int] = None,
    max_wallclock: int = 600,
):
    run_python_file.remote(
        script_path=file,
        script_args=[],
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        run_id=run_id,
        iterations=iterations,
        max_wallclock=max_wallclock,
    )

