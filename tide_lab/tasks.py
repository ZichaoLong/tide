"""Project-owned delayed associative retrieval with disjoint mapping splits.

Each association is one atomic token (key, value), followed by delay tokens and
one query-key token. The answer is a value class, never an input answer token.
Sequences are independent episodes; no state is carried between examples.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import random

import torch


@dataclass(frozen=True)
class RetrievalConfig:
    n_keys: int = 8
    n_values: int = 8
    pairs: int = 4
    delay: int = 2
    split_salt: str = "tide-associative-retrieval-v1"

    def __post_init__(self):
        if any(type(x) is not int for x in (self.n_keys, self.n_values, self.pairs, self.delay)):
            raise ValueError("task dimensions must be integers")
        if not (2 <= self.n_keys <= 64 and 2 <= self.n_values <= 64
                and 1 <= self.pairs <= self.n_keys and 0 <= self.delay <= 128):
            raise ValueError("invalid retrieval task dimensions")
        if not self.split_salt:
            raise ValueError("split_salt must not be empty")

    def to_dict(self):
        return asdict(self)


class RetrievalTask:
    """Stateless data batches; (split, seed, step) is the complete RNG cursor.

    The SHA256 of a sorted association map assigns 80/10/10 train/val/test.
    All queries and permutations of the same map stay in the same split.
    Different RNG seeds alone are deliberately not used as split protection.
    """
    def __init__(self, config=None):
        self.config = config if isinstance(config, RetrievalConfig) else RetrievalConfig(**(config or {}))
        c = self.config
        self.vocab_size = c.n_keys * c.n_values + c.n_keys + 1
        self.value_vocab_size = c.n_values
        self.sequence_length = c.pairs + c.delay + 1
        self.query_offset = c.n_keys * c.n_values
        self.filler_token = self.vocab_size - 1

    def to_dict(self):
        return dict(schema="tide-delayed-associative-retrieval-v1", **self.config.to_dict(),
                    vocab_size=self.vocab_size, value_vocab_size=self.value_vocab_size,
                    sequence_length=self.sequence_length,
                    split_policy="sha256(sorted association map,salt) modulo 100: train<80,val<90,test",
                    reset_policy="one fresh graph/LM state per sequence; full episode BPTT")

    def mapping_split(self, associations):
        canonical = json.dumps([self.config.split_salt, sorted(map(list, associations))],
                               separators=(",", ":")).encode()
        bucket = int.from_bytes(hashlib.sha256(canonical).digest()[:8], "big") % 100
        return "train" if bucket < 80 else "val" if bucket < 90 else "test"

    def decode(self, tokens):
        ids = tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
        c = self.config
        if len(ids) != self.sequence_length:
            raise ValueError("incorrect sequence length")
        pairs = [divmod(int(x), c.n_values) for x in ids[:c.pairs]]
        if any(not 0 <= k < c.n_keys for k, _ in pairs) or len({k for k, _ in pairs}) != c.pairs:
            raise ValueError("invalid or duplicate association key")
        if ids[c.pairs:-1] != [self.filler_token] * c.delay:
            raise ValueError("invalid delay tokens")
        query = int(ids[-1]) - self.query_offset
        if query not in dict(pairs):
            raise ValueError("query key is absent from associations")
        return pairs, query, dict(pairs)[query]

    def batch(self, split, batch_size, step, seed, device="cpu"):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        if any(type(x) is not int or x < 0 for x in (batch_size, step, seed)) or batch_size < 1:
            raise ValueError("positive batch_size and nonnegative step/seed required")
        raw = json.dumps([self.config.to_dict(), split, seed, step], sort_keys=True).encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(raw).digest(), "big"))
        c = self.config
        examples, targets = [], []
        attempts = 0
        while len(examples) < batch_size:
            attempts += 1
            if attempts > max(10000, batch_size * 1000):
                raise RuntimeError("bounded rejection sampling exhausted for this task split")
            keys = rng.sample(range(c.n_keys), c.pairs)
            values = [rng.randrange(c.n_values) for _ in keys]
            pairs = list(zip(keys, values))
            if self.mapping_split(pairs) != split:
                continue
            query_index = rng.randrange(c.pairs)
            examples.append([k * c.n_values + v for k, v in pairs]
                            + [self.filler_token] * c.delay + [self.query_offset + keys[query_index]])
            targets.append(values[query_index])
        return (torch.tensor(examples, dtype=torch.long, device=device),
                torch.tensor(targets, dtype=torch.long, device=device))


class NoMemoryClassifier(torch.nn.Module):
    """Query-only learned control: cannot inspect any stored association."""
    def __init__(self, vocab_size, value_vocab_size, width=16, seed=7, **unused):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = torch.nn.Embedding(vocab_size, width)
            self.head = torch.nn.Linear(width, value_vocab_size)
        self.last_metrics = {}

    def forward(self, token_ids):
        self.last_metrics = {"work/query_tokens": token_ids.shape[0], "work/history_tokens": 0}
        return self.head(self.embedding(token_ids[:, -1]))
