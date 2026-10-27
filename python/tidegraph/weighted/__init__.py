"""Pinned weighted-message local semantics; see docs/weighted-message.md."""
from .allocation import Allocation, allocate_quad
from .entropy_allocation import allocate
from .sending import attention_shares, attention_send, proportional, total_zero, hard_selection
from .binding import Binding, bind
from .config import KVConfig, RegionalConfig
from .recurrent_config import RecurrentConfig
from .convolution import ConvCache, FixedTimeConv
from .tensors import Payload, Received
from .source_binding import SourceBinding
from .topology import fully_connected, hb_lattice
from .instances import bind_layered, layered_inputs, readout
from .resident import WeightedResident
from .resident_training import WeightedResidentTraining

__all__ = ["Allocation", "allocate", "allocate_quad", "Binding", "bind", "KVConfig", "RegionalConfig", "SourceBinding",
           "attention_shares", "attention_send", "proportional", "total_zero", "hard_selection",
           "RecurrentConfig", "ConvCache", "FixedTimeConv", "Payload", "Received",
           "fully_connected", "hb_lattice", "bind_layered", "layered_inputs", "readout", "WeightedResident", "WeightedResidentTraining"]
