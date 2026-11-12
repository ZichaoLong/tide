#pragma once
#include "kernel_operator.h"

// One ordered metadata worker. Numerical stages consume the selected block;
// they are not scalar work performed by this scheduler.
namespace tide_block_dispatch_device {
using I = int64_t;
enum : I { Pending = 0, Chosen = 1, Complete = 2, Absent = 3 };
enum : int32_t { Invalid = 70, Capacity = 71, Unclosed = 72, Stale = 73 };

__aicore__ inline bool close(__gm__ I* edges, I edge_count, __gm__ I* state,
                            I source, bool emitted, __gm__ int32_t* error) {
  for (I e = 0; e < edge_count; ++e) if (edges[3*e] == source) {
    const I target = edges[3*e+1];
    if (state[3*target] != Pending || state[3*target+1] <= 0) {
      error[0] = Invalid; return false;
    }
    --state[3*target+1];
    if (emitted && edges[3*e+2]) ++state[3*target+2];
  }
  return true;
}

__aicore__ inline void reset(__gm__ I* tasks, __gm__ I* edges,
    __gm__ uint8_t* valid, __gm__ uint8_t* seeds, __gm__ I* epoch,
    __gm__ I* state, __gm__ I* ledger, __gm__ int32_t* error,
    I slots, I edge_count, I groups) {
  if (error[0]) return;
  if (epoch[0] <= ledger[0] || epoch[0] < 1) { error[0] = Stale; return; }
  // Validate every descriptor before publishing a new generation.
  for (I v = 0; v < slots; ++v) if (valid[v]) {
    if (tasks[5*v] < 0 || tasks[5*v] >= groups || tasks[5*v+1] < 0 ||
        tasks[5*v+2] < 0 || tasks[5*v+3] < 0 ||
        (tasks[5*v+4] != 0 && tasks[5*v+4] != 1)) { error[0] = Invalid; return; }
  }
  for (I e = 0; e < edge_count; ++e) {
    const I s = edges[3*e], t = edges[3*e+1], message = edges[3*e+2];
    if (s < 0 || s >= slots || t < 0 || t >= slots || s == t ||
        !valid[s] || !valid[t] || (message != 0 && message != 1)) {
      error[0] = Invalid; return;
    }
  }
  for (I v = 0; v < slots; ++v) {
    state[3*v] = valid[v] ? Pending : -1;
    state[3*v+1] = 0; state[3*v+2] = valid[v] && seeds[v] ? 1 : 0;
  }
  for (I e = 0; e < edge_count; ++e) ++state[3*edges[3*e+1]+1];
  ledger[0] = epoch[0]; ledger[1] = ledger[2] = ledger[3] = 0;
}

__aicore__ inline void choose(__gm__ I* tasks, __gm__ I* edges,
    __gm__ I* epoch, __gm__ I* state, __gm__ I* proposal,
    __gm__ I* selected, __gm__ I* meta, __gm__ I* ledger,
    __gm__ int32_t* branch, __gm__ int32_t* error,
    I slots, I edge_count, I groups, I rows, I journal_capacity) {
  branch[0] = 0; meta[0] = 0; meta[1] = -1; meta[2] = epoch[0]; meta[3] = 0;
  for (I r = 0; r < rows; ++r) selected[r] = slots; // Safe gather sentinel.
  if (error[0]) return;
  if (epoch[0] != ledger[0]) { error[0] = Stale; return; }
  for (I v = 0; v < slots; ++v) {
    const I status = state[3*v];
    if (status < -1 || status == Chosen || status > Absent || state[3*v+1] < 0 ||
        state[3*v+2] < 0) { error[0] = Invalid; return; }
  }
  for (I j = 0; j < slots*3; ++j) proposal[j] = state[j];
  // Absence closes a causal dependency too. It never fabricates a zero message.
  // Closure happens in scratch: numerical/capacity failure publishes no prefix.
  bool changed = true;
  while (changed) {
    changed = false;
    for (I v = 0; v < slots; ++v)
      if (proposal[3*v] == Pending && proposal[3*v+1] == 0 &&
          proposal[3*v+2] == 0 && !tasks[5*v+4]) {
        proposal[3*v] = Absent; ++meta[3]; changed = true;
        if (!close(edges, edge_count, proposal, v, false, error)) return;
      }
  }
  I group = -1, pending = 0;
  for (I v = 0; v < slots; ++v) if (proposal[3*v] == Pending) {
    ++pending;
    if (proposal[3*v+1] == 0 && group < 0) group = tasks[5*v];
  }
  if (group < 0) { if (pending) error[0] = Unclosed; return; }
  if (group >= groups) { error[0] = Invalid; return; }
  for (I v = 0; v < slots && meta[0] < rows; ++v)
    if (proposal[3*v] == Pending && proposal[3*v+1] == 0 && tasks[5*v] == group) {
      selected[meta[0]++] = v; proposal[3*v] = Chosen;
    }
  if (ledger[2] < 0 || ledger[2] > journal_capacity || meta[0] > journal_capacity-ledger[2]) {
    error[0] = Capacity; return;
  }
  meta[1] = group; branch[0] = static_cast<int32_t>(group+1);
}

__aicore__ inline void commit(__gm__ I* tasks, __gm__ I* edges,
    __gm__ I* epoch, __gm__ I* state, __gm__ I* proposal,
    __gm__ I* selected, __gm__ uint8_t* emitted, __gm__ I* meta,
    __gm__ I* ledger, __gm__ I* journal, __gm__ int32_t* branch,
    __gm__ int32_t* error, I slots, I edge_count, I rows, I journal_capacity) {
  branch[0] = 0;
  if (error[0]) return;
  if (meta[2] != epoch[0] || epoch[0] != ledger[0]) { error[0] = Stale; return; }
  const I count = meta[0], offset = ledger[2];
  if (count < 0 || count > rows || offset < 0 || offset > journal_capacity ||
      count > journal_capacity-offset || meta[3] < 0) { error[0] = Capacity; return; }
  for (I r = 0; r < count; ++r) {
    const I v = selected[r];
    if (v < 0 || v >= slots || proposal[3*v] != Chosen ||
        tasks[5*v] != meta[1] || emitted[r] > 1) { error[0] = Invalid; return; }
    for (I k = 0; k < r; ++k) if (selected[k] == v) { error[0] = Invalid; return; }
  }
  for (I r = 0; r < count; ++r) {
    const I v = selected[r]; proposal[3*v] = Complete;
    if (!close(edges, edge_count, proposal, v, emitted[r] != 0, error)) return;
  }
  // Commit is reached only after all numerical guards and structural checks.
  for (I j = 0; j < slots*3; ++j) state[j] = proposal[j];
  for (I r = 0; r < count; ++r) {
    journal[3*(offset+r)] = selected[r]; journal[3*(offset+r)+1] = emitted[r];
    journal[3*(offset+r)+2] = ledger[1];
  }
  ledger[1] += count > 0 ? 1 : 0; ledger[2] += count; ledger[3] += meta[3];
  meta[2] = -1; // A completion token is consumed, including an empty closure.
  branch[0] = count > 0 ? 1 : 0;
}
}
