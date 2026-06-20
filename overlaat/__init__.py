"""Overlaat: fair-queueing + usage accounting sidecar for self-hosted LLM gateways."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("overlaat")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"
