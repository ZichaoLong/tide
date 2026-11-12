#pragma once
// 0770593 Settle §10 / weighted raw state: exact complete-position owners.
// This adapter only decodes computed tensors. It never replays local equations.
#include "tide/weighted_sequence_columns.h"

namespace tide::weighted::columns {
void validate_owner(const SettleGraph&,const Model&,const Continuation&,Index samples);
std::map<Index,LocalState> import_state(const SettleGraph&,const Model&,const Continuation&);
Continuation export_state(const SettleGraph&,const Model&,const Result&,
                          const std::map<Index,Message>& ports,Index stop_position,
                          const Continuation& previous);

struct WindowResult {
  Result columns;
  Continuation continuation;
};
// Owns fixed graph/module/placement handles, accepts an explicit raw cursor.
// Parameters may be updated in-place by an optimizer between detached windows.
// Rebuild after replacing handles or changing placement/configuration. Ordinary
// column roots remain differentiable across calls until the caller detaches.
class WindowExecutor {
  SettleGraph spec_;
  Model model_;
  std::optional<ExecutionPlacement> placement_;
  Index max_workspace_bytes_;
 public:
  WindowExecutor(SettleGraph,Model,Index max_workspace_bytes=Index{4}*1024*1024*1024,
                 std::optional<ExecutionPlacement> placement=std::nullopt);
  WindowResult run_ports(const Continuation&,const Layout&,
                         const std::map<Index,Message>&,Index stop_position) const;
  const SettleGraph& spec() const { return spec_; }
};
}
