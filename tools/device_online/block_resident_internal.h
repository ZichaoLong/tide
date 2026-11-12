#pragma once
#include "block_settle_model.h"
#include "block_settle_observations.h"
#include "block_settle_session_internal.h"
namespace tide::device_online {
BlockSettleCheckpoint import_block_checkpoint(const BlockCheckpoint&);
BlockCheckpoint export_block_checkpoint(const BlockSettleCheckpoint&);
struct BlockResidentSession::Impl {
  struct Window {BlockWindow view;BlockSettleWindow actual;std::vector<block_session_detail::Version> versions;};
  SettleGraph graph;Model model;BlockSettleModel binding;
  std::unique_ptr<BlockSettleSession> owner;
  uint64_t session;Index batch,cut,generation=0;
  ResidentOptimizerKind kind;
  std::vector<OptimizerGroup> groups;
  BlockResidentLimits limits;
  std::vector<Window> windows;
  bool failed=false,closed=false;
  Impl(SettleGraph,Model,const Continuation&,at::Device,ResidentOptimizerKind,
    std::vector<OptimizerGroup>,BlockResidentLimits,const BlockCheckpoint*);
  void check()const;
};
}
