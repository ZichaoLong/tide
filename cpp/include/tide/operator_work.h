#pragma once
// Optional process-wide forward accounting. Enable/reset/snapshot only with
// workers quiescent. One measured executor per process; no backward/replay claim.
// No tensor reads, clocks, or callbacks on the hot path. FMA counts as 2 FLOPs.
#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <string>

namespace tide::work {
using Count = std::int64_t;
enum Field { QkvCalls, QkvRows, QkvFlops, OutCalls, OutRows, OutFlops,
  EmitCalls, EmitRows, EmitFlops, HeadCalls, HeadRows, HeadFlops,
  ValidScores, ExecutedScores, ValidAttentionFlops, ExecutedAttentionFlops,
  AttentionCalls, EmitEdgeRows, PendingEdgeRows, BodyCandidates, BodySelected,
  AggregateScaleElements, AggregateAddElements, Fields };
inline constexpr std::array<const char*, Fields> names{
  "qkv_calls", "qkv_rows", "qkv_flops", "out_calls", "out_rows", "out_flops",
  "emit_calls", "emit_rows", "emit_flops", "head_calls", "head_rows", "head_flops",
  "valid_score_elements", "executed_score_elements", "valid_attention_flops",
  "executed_attention_flops", "attention_calls", "emit_edge_rows", "pending_edge_rows",
  "body_candidates", "body_selected", "aggregate_scale_elements", "aggregate_add_elements"};
inline std::atomic<bool> active{false};
inline std::array<std::atomic<Count>, Fields> counts{};
inline bool enabled() { return active.load(std::memory_order_relaxed); }
inline void reset(bool on) {
  active.store(false, std::memory_order_relaxed);
  for (auto& n : counts) n.store(0, std::memory_order_relaxed);
  active.store(on, std::memory_order_relaxed);
}
inline void add(Field field, Count n) { counts[field].fetch_add(n, std::memory_order_relaxed); }
inline void linear(Field first, Count rows, Count in, Count out) {
  if (!enabled()) return;
  add(first, 1); add(Field(first+1), rows); add(Field(first+2), 2*rows*in*out);
}
inline void attention(Count width, Count heads, Count valid_pairs, Count executed_pairs) {
  if (!enabled()) return;
  add(AttentionCalls, 1); add(ValidScores, valid_pairs*heads);
  add(ExecutedScores, executed_pairs*heads);
  add(ValidAttentionFlops, 4*valid_pairs*width);
  add(ExecutedAttentionFlops, 4*executed_pairs*width);
}
inline void emit(Count rows, Count width, Count edges, Count pending_rows = 0) {
  if (!enabled()) return;
  if (edges) linear(EmitCalls, rows, width, edges*width);
  add(EmitEdgeRows, rows*edges); add(PendingEdgeRows, pending_rows*edges);
}
inline std::map<std::string, double> metrics() {
  std::map<std::string, double> result;
  for (int i = 0; i < Fields; ++i) result[std::string("op/")+names[i]] = counts[i].load();
  result["op/linear_flops"] = result["op/qkv_flops"]+result["op/out_flops"]+
    result["op/emit_flops"]+result["op/head_flops"];
  result["op/executed_matmul_flops"] = result["op/linear_flops"]+result["op/executed_attention_flops"];
  return result;
}
} // namespace tide::work
