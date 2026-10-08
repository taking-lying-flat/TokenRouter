import os

import torch
import torch.nn as nn
from torchvision.ops import MLP


_ROUTER_CANDIDATES = (
    "confidence_model.pth",
    "router.pth",
    "router.pt",
)


def resolve_router_path(router_path: str) -> str:
    """Resolve a CITER router from a local path or Hugging Face repo."""
    if os.path.isfile(router_path):
        return router_path

    if os.path.isdir(router_path):
        for filename in _ROUTER_CANDIDATES:
            candidate = os.path.join(router_path, filename)
            if os.path.isfile(candidate):
                return candidate

        checkpoint_files = sorted(
            name
            for name in os.listdir(router_path)
            if name.endswith((".pth", ".pt"))
            and os.path.isfile(os.path.join(router_path, name))
        )
        if checkpoint_files:
            return os.path.join(router_path, checkpoint_files[0])

        raise FileNotFoundError(
            f"No CITER router checkpoint found in local directory: {router_path}"
        )

    if os.path.isabs(router_path):
        raise FileNotFoundError(f"CITER router checkpoint does not exist: {router_path}")

    hf_error = None
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        repo_id, filename = _split_hf_router_path(router_path)
        if filename is None:
            repo_files = list_repo_files(repo_id)
            filename = next(
                (name for name in _ROUTER_CANDIDATES if name in repo_files),
                None,
            )
            if filename is None:
                checkpoint_files = sorted(
                    name for name in repo_files if name.endswith((".pth", ".pt"))
                )
                if checkpoint_files:
                    filename = checkpoint_files[0]

        if filename is not None:
            return hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as exc:
        hf_error = exc

    message = f"Unable to resolve CITER router path: {router_path}"
    if hf_error is not None:
        message += f" (Hugging Face lookup failed: {hf_error})"
    raise FileNotFoundError(message)


def _split_hf_router_path(router_path: str) -> tuple[str, str | None]:
    parts = router_path.strip("/").split("/")
    if len(parts) >= 3 and parts[-1].endswith((".pth", ".pt")):
        return "/".join(parts[:2]), "/".join(parts[2:])
    return router_path, None


class ConfidenceModel(nn.Module):
    def __init__(self, config, threshold, device):
        super().__init__()
        self.config = config
        self.mlp = MLP(
            in_channels=config.hidden_size,
            hidden_channels=[2 * config.hidden_size, config.hidden_size],
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU,
            dropout=0.1,
        ).to(device=device, dtype=config.dtype)
        self.linear = nn.Linear(
            config.hidden_size, 1, device=device, dtype=config.dtype
        )
        self.sigmoid = nn.Sigmoid().to(device=device, dtype=config.dtype)
        self.threshold = threshold

    def forward(self, hidden_states, token_indices=None) -> torch.Tensor:

        # reshape to batch_size * seq_length, hidden_size
        batch_size, seq_length, hidden_size = hidden_states.size()
        mlp_input = hidden_states.view(-1, hidden_size)

        # pass all embeddings to mlp
        mlp_output = self.mlp(mlp_input)
        linear_output = self.linear(mlp_output)

        # reshape to batch_size, seq_length, 1
        scores = linear_output.view(batch_size, seq_length, -1)

        if token_indices is not None:
            # extract the score for the specified token indices
            confidence_scores = torch.gather(
                scores, 1, token_indices.unsqueeze(-1)
            ).squeeze(-1)
        else:
            confidence_scores = scores

        return self.sigmoid(confidence_scores) < self.threshold
