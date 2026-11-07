"""Pretrained Qwen task adaptation with one bounded residual branch.

The language model is an installed Transformers model loaded from a fixed local
checkpoint. Only the task readout, symbolic token mapping and residual branch
belong to this project. The TIDE branch uses Core5Encoder's public foundation
API; no execution implementation is copied here.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import string

import torch


CHECKPOINT_REPOSITORY = "Qwen/Qwen2.5-0.5B-Instruct"
CHECKPOINT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
CHECKPOINT_FILES = {
    "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
}


def verify_checkpoint(path):
    """Reject a different checkpoint/tokenizer before emitting fixed ancestry."""
    for filename, expected in CHECKPOINT_FILES.items():
        with (path / filename).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"selected checkpoint identity mismatch: {filename}")
    return dict(CHECKPOINT_FILES)


def symbolic_token_map(tokenizer, vocab_size):
    """Map task symbols injectively to original, individually encoded tokens.

    This is an arbitrary fixed codebook, not a natural-language rendering of
    key/value pairs. No new input embeddings or tokenizer tokens are added.
    """
    choices = string.ascii_uppercase + string.ascii_lowercase + string.digits + string.punctuation
    ids, symbols = [], []
    for symbol in choices:
        encoded = tokenizer.encode(symbol, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] not in ids and encoded[0] not in tokenizer.all_special_ids:
            ids.append(encoded[0])
            symbols.append(symbol)
        if len(ids) == vocab_size:
            return ids, symbols
    raise ValueError(f"only {len(ids)} distinct single-token symbols, requested {vocab_size}")


@dataclass
class GrowthInferenceState:
    """Inference-only state. Core5 is explicitly replayed over hidden history.

    The HF backbone uses its KV cache; the tiny branch replays the accumulated
    features instead of claiming an optimized joint streaming implementation.
    This object is not an application checkpoint or a training continuation.
    """
    past_key_values: object
    hidden_history: torch.Tensor


class GrowthClassifier(torch.nn.Module):
    def __init__(self, checkpoint_path, vocab_size=73, value_vocab_size=8,
                 branch="none", seed=7, width=16, topology="fc", depth=2,
                 region_width=4, send="dense", freeze_backbone=False):
        super().__init__()
        if branch not in {"none", "tide", "mlp"}:
            raise ValueError("branch must be none, tide or mlp")
        checkpoint_path = Path(checkpoint_path).resolve(strict=True)
        if not (checkpoint_path / "model.safetensors").is_file():
            raise ValueError("expected a complete local safetensors checkpoint")
        self.checkpoint_files = verify_checkpoint(checkpoint_path)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True,
                                                      trust_remote_code=False)
        self.lm = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32, attn_implementation="eager")
        if self.lm.config.model_type != "qwen2":
            raise ValueError("first batch is fixed to the selected Qwen2 checkpoint family")
        self.lm.config.use_cache = False
        self.branch = branch
        self.freeze_backbone = bool(freeze_backbone)
        self.vocab_size, self.value_vocab_size = vocab_size, value_vocab_size
        self.width = width
        ids, symbols = symbolic_token_map(self.tokenizer, vocab_size)
        self.register_buffer("token_map", torch.tensor(ids, dtype=torch.long))
        self.symbols = symbols
        hidden_size = self.lm.config.hidden_size
        # The shared head is initialized before branch construction. Forking
        # prevents branch choice from changing the caller's subsequent RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.head = torch.nn.Linear(hidden_size, value_vocab_size)
            if branch != "none":
                self.branch_input = torch.nn.Linear(hidden_size, width, bias=False)
                if branch == "tide":
                    from .core5 import Core5Encoder
                    self.branch_core = Core5Encoder(width=width, topology=topology,
                        depth=depth, region_width=region_width, send=send, seed=seed)
                else:
                    self.branch_core = torch.nn.Sequential(
                        torch.nn.Linear(width, width), torch.nn.GELU(),
                        torch.nn.Linear(width, width))
                self.branch_output = torch.nn.Linear(width, hidden_size, bias=False)
                torch.nn.init.zeros_(self.branch_output.weight)
        if self.freeze_backbone:
            self.lm.requires_grad_(False)
        self.last_metrics = {}

    def map_tokens(self, token_ids):
        if token_ids.ndim != 2 or token_ids.shape[1] < 1:
            raise ValueError("task tokens must have shape [batch, nonempty positions]")
        return self.token_map[token_ids]

    def _hidden(self, input_ids, *, use_cache=False, past_key_values=None):
        # Do not compute all-vocabulary LM logits for the synthetic classifier.
        return self.lm.model(input_ids=input_ids, use_cache=use_cache,
                             past_key_values=past_key_values, return_dict=True)

    def residual(self, hidden, *, branch_enabled=True):
        if self.branch == "none" or not branch_enabled:
            self.last_metrics = {"branch/enabled": 0}
            return torch.zeros_like(hidden[:, -1])
        features = torch.tanh(self.branch_input(hidden))
        if self.branch == "tide":
            value = self.branch_core(features)
            self.last_metrics = dict(self.branch_core.last_metrics)
        else:
            value = self.branch_core(features[:, -1])
            self.last_metrics = {"work/branch_tokens": hidden.shape[0]}
        self.last_metrics["branch/enabled"] = 1
        return self.branch_output(value)

    def forward(self, token_ids, *, branch_enabled=True):
        hidden = self._hidden(self.map_tokens(token_ids)).last_hidden_state
        return self.head(hidden[:, -1] + self.residual(hidden, branch_enabled=branch_enabled))

    def language_last_logits(self, input_ids, *, branch_enabled=True):
        """Last-position LM logits on ordinary tokenizer input, for calibration.

        The task-training loss uses the separate classification head. Language
        quality after adaptation is not inferred from synthetic task accuracy.
        """
        hidden = self._hidden(input_ids).last_hidden_state
        return self.lm.get_output_embeddings()(
            hidden[:, -1] + self.residual(hidden, branch_enabled=branch_enabled))

    @torch.no_grad()
    def infer_tokens(self, token_ids, state=None, *, branch_enabled=True):
        """Cached backbone + replayed branch; accepts one or several new tokens."""
        if self.training:
            raise RuntimeError("inference continuation requires eval mode")
        if state is not None and state.hidden_history.shape[0] != token_ids.shape[0]:
            raise ValueError("continuation batch size changed")
        result = self._hidden(self.map_tokens(token_ids), use_cache=True,
                              past_key_values=None if state is None else state.past_key_values)
        hidden = result.last_hidden_state
        if state is not None:
            hidden = torch.cat((state.hidden_history, hidden), dim=1)
        value = self.head(hidden[:, -1] + self.residual(hidden, branch_enabled=branch_enabled))
        return value, GrowthInferenceState(result.past_key_values, hidden.detach())

    def parameter_summary(self):
        return dict(
            backbone_parameters=sum(p.numel() for p in self.lm.parameters()),
            task_head_parameters=sum(p.numel() for p in self.head.parameters()),
            branch_parameters=sum(p.numel() for name, p in self.named_parameters()
                                  if name.startswith("branch_")),
            total_parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            tied_input_output_embeddings=(self.lm.get_input_embeddings().weight is
                                         self.lm.get_output_embeddings().weight))

    def facts(self):
        facts = dict(self.parameter_summary(), checkpoint_repository=CHECKPOINT_REPOSITORY,
            checkpoint_revision=CHECKPOINT_REVISION, checkpoint_files=self.checkpoint_files,
            branch=self.branch,
            backbone_training="frozen" if self.freeze_backbone else "all parameters",
            symbolic_mapping=dict(ids=self.token_map.detach().cpu().tolist(), symbols=self.symbols,
                semantics="arbitrary atomic task symbol to original single-token codebook"),
            initial_neutrality="zero trainable output projection; inner gradients start at zero",
            branch_input="final backbone hidden sequence, linear projection and tanh",
            branch_readout="last position residual before task classification head",
            continuation="cached backbone plus full hidden-prefix replay for branch",
            training_state="fresh per episode; full episode BPTT; no cache across updates",
            lm_attention="eager", dtype="float32")
        if self.branch == "tide":
            facts["core5"] = self.branch_core.facts()
        return facts
