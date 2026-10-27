"""Pinned recurrent-memory §S: dimensions and applicable configuration domains."""
from dataclasses import dataclass
from .config import finite, positive_int


@dataclass(frozen=True)
class RecurrentConfig:
    memory_family: str = "affine"
    width: int = 4
    summary_width: int = 2
    key_width: int = 2
    value_width: int = 3
    ffn_width: int = 8
    memory_read: str | None = None
    lambda_short: float = .1
    epsilon: float = 1e-5
    alpha_mem: float = 1.
    alpha_ff: float = 1.
    heads: int = 1
    groups: int = 1
    matrix_rates: tuple[float, ...] = ()
    conv_size: int = 1
    conv_stride: int = 1
    qk_epsilon: float = 1e-6
    output_epsilon: float = 1e-5

    def __post_init__(self):
        if self.memory_family not in {"affine", "delta", "gated_delta", "ssd"}:
            raise ValueError("unknown recurrent memory_family")
        for name in ("width", "summary_width", "key_width", "value_width", "ffn_width",
                     "heads", "groups", "conv_size", "conv_stride"):
            positive_int(name, getattr(self, name))
        for name in ("lambda_short", "epsilon", "qk_epsilon", "output_epsilon"):
            finite(name, getattr(self, name), 0, True)
        for name in ("alpha_mem", "alpha_ff"):
            finite(name, getattr(self, name), 0)
        mode = ("sum" if self.basic else "state") if self.memory_read is None else self.memory_read
        if mode not in {"sum", "state"}:
            raise ValueError("recurrent memory_read must be sum/state")
        object.__setattr__(self, "memory_read", mode)
        rates = tuple(self.matrix_rates)
        if self.basic:
            if (self.heads, self.groups, self.conv_size, self.conv_stride) != (1, 1, 1, 1) or rates:
                raise ValueError("basic recurrence has two time scales, no heads/groups/convolution")
        else:
            rates = rates or (0.,)*self.heads
            if len(rates) != self.heads:
                raise ValueError("one passive matrix rate is required per head")
        if self.memory_family != "ssd" and self.groups != 1 or self.heads % self.groups:
            raise ValueError("groups apply only to SSD and must divide heads")
        for value in rates:
            finite("matrix rate", value, 0)
        if self.memory_family != "gated_delta" and self.qk_epsilon != 1e-6:
            raise ValueError("qk_epsilon applies only to gated_delta")
        if self.basic and self.output_epsilon != 1e-5:
            raise ValueError("output_epsilon applies only to gated_delta/ssd")
        positive_int("convolution capacity", (self.conv_size-1)*self.conv_stride+1)
        positive_int("matrix capacity", (2 if self.basic else self.heads)*self.key_width*self.value_width)
        if not self.basic:
            positive_int("convolution width", self.conv_width)
            positive_int("projection width", self.heads*self.value_width+self.conv_width+self.heads)
        object.__setattr__(self, "matrix_rates", rates)

    @property
    def basic(self):
        return self.memory_family in {"affine", "delta"}

    @property
    def matrix_shape(self):
        return (2 if self.basic else self.heads, self.value_width, self.key_width)

    @property
    def conv_width(self):
        if self.basic:
            return 0
        if self.memory_family == "gated_delta":
            return self.heads*(2*self.key_width+self.value_width)
        return self.heads*self.value_width+2*self.groups*self.key_width
