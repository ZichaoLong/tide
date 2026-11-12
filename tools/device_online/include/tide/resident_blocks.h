#pragma once
// Public Model-bound tensor interface for the independent block executor.
// Explicit first-order maps: consumer heads/losses remain outside this owner.
#include "tide/settle.h"
#include "tide/resident_training.h"
namespace tide::device_online {
struct BlockResidentLimits {
  // Whole-call forward/reverse tensor increments, actual optimizer+step
  // inventory, and unique live tensors+workspaces (no fixed child fractions).
  int64_t cache=64,forward_bytes=512LL<<20,backward_bytes=1024LL<<20;
  int64_t optimizer_bytes=128LL<<20,total_bytes=3LL<<30,workspace_bytes=512LL<<20,windows=4;
  int32_t timeout_ms=30000;
};
struct BlockField {Tensor values,present,valid;};
struct BlockRoot {Tensor values,connected;};
struct BlockWindow {
  ResidentToken token;
  std::map<std::string,BlockField> fields;
  std::map<std::string,Tensor> metadata;
};
struct BlockRoots {ResidentToken token;std::map<std::string,BlockRoot> fields;};
struct BlockGradients {
  ResidentParameterGradient parameters;
  std::vector<BlockRoot> inputs;
  std::map<std::string,BlockRoot> initial,bound_initial;
};
struct BlockRawRecord {Tensor floating,integer;};
struct BlockCheckpoint {
  int64_t schema=2,generation=0,next_token=0;
  std::string identity;
  std::vector<std::vector<std::string>> aliases;
  std::map<std::string,Tensor> parameters;
  ResidentOptimizerKind optimizer=ResidentOptimizerKind::adamw;
  std::vector<OptimizerGroup> groups;
  ResidentOptimizerState slots;
  std::vector<BlockRawRecord> state,bound_initial;
  Tensor last_positions,node_present,region_times,input_counts,input_times;
};
// Capabilities are checked from installed mathematical bindings. This interface
// needs an actual Settle certificate; an arbitrary DAG is never guessed aligned.
// Tensor fields are batches, raw clocks are int64 and physical ports stay distinct.
class BlockResidentSession {
 public:
  BlockResidentSession(SettleGraph,Model,const Continuation&,at::Device,
    ResidentOptimizerKind = ResidentOptimizerKind::adamw,std::vector<OptimizerGroup> = {},BlockResidentLimits = {});
  BlockResidentSession(SettleGraph,Model,const Continuation&,const BlockCheckpoint&,at::Device,BlockResidentLimits={});
  ~BlockResidentSession();
  BlockWindow advance(const Tensor& int64_positions,const Tensor& payload,const Tensor& present);
  BlockGradients backward(const std::vector<BlockRoots>&);
  void step();
  void detach();
  BlockCheckpoint checkpoint()const;
  Continuation continuation()const; // Detached complete cut, actual raw/history/ledger decode.
  Index cut()const;
  Index generation()const;
  ResidentOptimizerKind optimizer_kind()const;
  std::vector<OptimizerGroup> optimizer_groups()const;
  void close();
 private:
  struct Impl;std::unique_ptr<Impl> impl_;
};
void save_block_checkpoint(const BlockCheckpoint&,const std::string&,int64_t max_bytes=int64_t{1}<<32);
BlockCheckpoint load_block_checkpoint(const std::string&,int64_t max_bytes=int64_t{1}<<32);
}
