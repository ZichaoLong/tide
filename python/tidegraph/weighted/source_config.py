"""Architecture §2.2 parameter-sharing configuration; no tensor allocation."""
from dataclasses import dataclass
from .config import positive_int


@dataclass(frozen=True)
class SourceBinding:
    mode: str = "region"
    rank: int | None = None

    def __post_init__(self):
        if self.mode not in {"region", "node", "source", "adapted"}:
            raise ValueError("unknown source parameter binding")
        if self.mode == "adapted":
            positive_int("source adapter rank", self.rank)
        elif self.rank is not None:
            raise ValueError("source adapter rank is only applicable to adapted binding")

