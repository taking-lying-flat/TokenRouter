"""Launch a stock SGLang server with local monkey patches installed."""

import os
import sys

from tokenrouter.sglang_hidden_states_patch import apply_sglang_hidden_states_patch

apply_sglang_hidden_states_patch()

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
