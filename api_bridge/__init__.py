BRIDGE_PROTOCOL_VERSION = 1

from .resource_registry import ResourceRegistry, get_resource_registry
from .runtime_registry import RuntimeRegistry, get_runtime_registry

__all__ = [
    "BRIDGE_PROTOCOL_VERSION",
    "ResourceRegistry",
    "RuntimeRegistry",
    "get_resource_registry",
    "get_runtime_registry",
]
