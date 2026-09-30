#pragma once
#include "kernel_operator.h"

namespace tide_device {
// Event rows are grouped by cache owner, then increasing logical time:
// [ready row, owner, parameter, old length, proposed length,
//  initial length + preceding event count, first event of this owner].
// Staged KV is [all original owner arenas, one compact row per actual event,
// zero sentinel]. A window changes visibility, never the underlying stage.
__aicore__ inline int64_t event_key_row(__gm__ int64_t* events,int64_t event,
    int64_t key,int64_t capacity,int64_t owners,bool proposed=true) {
  const auto row=events+event*7;
  const int64_t first=row[6],initial=row[5]-(event-first);
  const int64_t start=row[5]+int64_t(proposed)-row[3+int64_t(proposed)];
  const int64_t position=start+key;
  return position<initial?row[1]*capacity+position:
    owners*capacity+first+position-initial;
}
} // namespace tide_device
