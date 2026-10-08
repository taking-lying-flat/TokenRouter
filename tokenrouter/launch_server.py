import argparse
import os

from sglang.srt.utils import kill_process_tree

from tokenrouter.entrypoints.http_server import launch_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument(
        "--local_model",
        type=str,
        default=None,
        help=(
            "Launch only this model from the shared config. Overrides "
            "system.local_model and enables configured TCP peer endpoints."
        ),
    )
    args = parser.parse_args()

    try:
        launch_server(args.config_path, local_model=args.local_model)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
