"""Compatibility wrapper: re-export TMOEncoder from tmo_block."""
from .tmo_block import TMOEncoder, TMOBlock

__all__ = ['TMOEncoder', 'TMOBlock']