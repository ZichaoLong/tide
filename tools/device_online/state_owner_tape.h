#pragma once
#include "state_reverse_view.h"

namespace tide::device_online {
// Local parameter/cache records, with a static inverse node map. Actual event
// and source journals are packed from the candidate's coordinator tape later;
// state.metadata/values/count are intentionally absent in this owner fragment.
struct StateOwnerTape {
  std::vector<int64_t> global_nodes;
  StateReverseLayout layout;
  StateTape state;
  at::Tensor read,read_modes,read_kinds,sources;
  std::vector<EventAttentionTape> attention;
  std::vector<FiberAttentionTape> fiber;
};
struct RetainedStateOwnerTape {StateOwnerTape tape;int64_t tensor_bytes;};
int64_t state_owner_tape_bytes(const StateOwnerTape&);
RetainedStateOwnerTape retain_state_owner_tape(const StateOwnerTape&,int64_t tensor_budget);
// Publication targets actual forward storage, never grouped fiber tape gathers.
struct StateOwnerBanks {
  std::vector<int64_t> global_nodes;
  at::Tensor decay,retention,read;
  std::vector<EventAttentionTape> attention;
  FiberParameterBanks fiber;
};
} // namespace tide::device_online
