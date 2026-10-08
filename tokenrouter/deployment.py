from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class PeerEndpoints:
    bind: str
    connect: str


def resolve_local_model(config: Mapping, override: Optional[str] = None) -> Optional[str]:
    """Return the model launched by this process.

    ``None`` preserves the legacy behavior where every configured model is
    launched in one process. A command-line override takes precedence over
    ``system.local_model`` so one shared config can be used on every node.
    """

    model_names = list(config.get("model", {}))
    local_model = override
    if local_model is None:
        local_model = config.get("system", {}).get("local_model")

    if local_model is not None and local_model not in model_names:
        raise ValueError(
            f"Local model '{local_model}' not found in model config. "
            f"Available models: {model_names}"
        )
    return local_model


def resolve_peer_endpoints(config: Mapping, local_model: str) -> PeerEndpoints:
    """Resolve this model's bind endpoint and its peer's advertised endpoint."""

    model_names = list(config.get("model", {}))
    peer_names = [name for name in model_names if name != local_model]
    if len(peer_names) != 1:
        raise ValueError(
            "Distributed TokenRouter currently requires exactly two configured "
            f"models; found {model_names}"
        )

    peer_configs = config.get("system", {}).get("peer_endpoints")
    if not isinstance(peer_configs, Mapping):
        raise ValueError(
            "system.peer_endpoints is required when local_model is set"
        )

    peer_model = peer_names[0]
    try:
        local_config = peer_configs[local_model]
        peer_config = peer_configs[peer_model]
        bind_endpoint = local_config["bind"]
        connect_endpoint = peer_config["advertise"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Each model in system.peer_endpoints must define 'bind' and "
            f"'advertise' endpoints; local_model={local_model!r}, "
            f"peer_model={peer_model!r}"
        ) from exc

    if not isinstance(bind_endpoint, str) or not bind_endpoint:
        raise ValueError(f"Invalid bind endpoint for model '{local_model}'")
    if not isinstance(connect_endpoint, str) or not connect_endpoint:
        raise ValueError(f"Invalid advertise endpoint for model '{peer_model}'")

    return PeerEndpoints(bind=bind_endpoint, connect=connect_endpoint)
