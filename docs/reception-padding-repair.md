# Reception structural membership repair — 2026-11-20

The downstream user authorized repair in this foundation after a real FP32 training failure. This change makes static topology membership mandatory in `TideModel._reception_mask`. Missing padded edges supply no derivative roots; genuine zero-mass edges keep their declared right derivative. A later routing multiplication by zero cannot remove an already overflowed padding derivative.

`reception_mask_padding` defaults to true. Explicit false remains readable and serializes unchanged for old checkpoints, but cannot disable membership. Parameters, state format and mathematical bindings are unchanged; effective execution and auto backend selection can change. Data loading, trajectory equivalence and downstream acceptance remain separate. Both ordinary and batched region paths use the same mask, including recomputation and placement.

The working source was byte-identical to the current downstream r04 `src/tide` before this patch. Preexisting allocation/Shannon changes and their documentation/tests were preserved by SHA256. No commit, push, environment change or active/frozen campaign edit occurred. Qualification used a frozen snapshot with the dirty source included at:

`/mi/data2T/zlong/home/llm/tide-learning-lab/.local/pilot-r3/padding-repair-r01/`

The `snapshot.json` records all 196 Python source/test files. Source SHA256:

- `src/tide/model.py`: `6e38b6d6e0382688cc8b7e01de8f87adff35e8777cbd9d41a87dd705c44e936d`
- `src/tide/config.py`: `01e4be6898e1d5a156ff29c2739509c30b65b372ff65c39584b3b4f1a72d520a`

## Checked scope

223 CPU directed tests passed: 156 primitive/range/Shannon checks and 67 reception/layout/batching/recompute/placement/recovery checks. Legacy false is exercised explicitly. Twenty-six directed NPU tests passed on one queued Ascend card. Original failed attention arguments passed independent CPU FP64 normalized-quotient/standard-autograd output and Q/K/V/mass VJPs on CPU and NPU under `atol=1e-5, rtol=1e-4`.

The actual failed CPU batch completes backward and Adam update after repair. Three NPU saved-batch updates plus a fresh-process replay load model/optimizer/private/RNG exactly; the following update has maximum difference 7.45e-9, within the original threshold. A real healthy downstream CPU checkpoint also loads RNG and data stream exactly, performs three next-batch updates, and replays the third in a fresh process bitwise exactly. Diagnostic prefixes are not campaign continuation checkpoints.

These checks do not certify every numerical path or model quality. At a healthy state, an old finite padding VJP of 1.57e23 disappeared with bitwise identical real gradients, parameters, optimizer and private state when both reception paths used tiled. Actual auto old/new comparisons pass tolerance but are not bitwise identical.

## Retained numerical limits

A full-model CPU FP64 precision comparison still has a private KV-state field outside the original threshold. Forcing Reception tiled adds three update-delta failures: approximately 1e-9 cancellation residuals at nearly zero Q/K gradients are amplified by Adam with initially zero moments into approximately 1e-5 differences. The sampled actual auto path has exact zero gradients at these three positions; the forced-path failure is not proof of identical pollution in the campaign. A separate single-real-key analytic witness reproduces nonzero tiled Q/K residuals where the literal derivative is zero. Attention/optimizer policies were not changed in this repair.

The independent local oracles and full-model precision diagnostic have different scope. Preserve the downstream Shannon restoration failures and all original thresholds. Follow-up should quantify actual auto fallback exposure and address the structural single-key identity before broad learning claims, without clipping valid small gradients.

The NPU probe completed and released its recorded descendants, cgroup and cooperative locks. The generic strict lifecycle audit retains two metadata binding omissions because its prepared adapter was generated after execution and the unused corpus manifest was not in the diagnostic source identity. A separate review verifies the prelaunch-bound actual plan/launcher/source and current-cycle data identity; the strict audit was not relabelled. No campaign restart or stage-B admission occurred.

Detailed consumer evidence and reproduction scripts are linked from [the current repair review](../../tide-learning-lab/docs/pilot-r3-reception-padding-repair.md) and [machine-readable evidence](../../tide-learning-lab/docs/evidence/pilot-r3-reception-padding-repair.json). This source delivery does not by itself migrate the frozen consumer installation.
