"""Pinned weighted-message local semantics; see docs/weighted-message.md."""
from .allocation import Allocation, allocate_quad
from .binding import Binding, bind
from .config import KVConfig, RegionalConfig
from .recurrent_config import RecurrentConfig
from .convolution import ConvCache, FixedTimeConv
from .tensors import Payload, Received
from .topology import fully_connected, hb_lattice
from .instances import bind_layered, layered_inputs, readout

__all__ = ["Allocation", "allocate_quad", "Binding", "bind", "KVConfig", "RegionalConfig",
           "RecurrentConfig", "ConvCache", "FixedTimeConv", "Payload", "Received",
           "fully_connected", "hb_lattice", "bind_layered", "layered_inputs", "readout"]
