#pragma once
#include "tide/resident.h"
#include "tide/optimizer.h"

namespace tide {
enum class ResidentOptimizerKind { sgd, adamw };
struct ResidentTrainingLimits {
  ResidentLimits forward;
  Index windows=8, retained_bytes=128*1024*1024, backward_bytes=512*1024*1024;
  Index optimizer_bytes=128*1024*1024, program_workspace_bytes=64*1024*1024;
  Index reverse_chunk_rows=16;
};
struct ResidentToken { uint64_t session=0; Index index=0, generation=0; };
struct ResidentTrainingWindow {
  ResidentToken token;
  Index start=0, stop=0;
  ResidentWindow outputs;
  Tensor pending_coordinates, pending_values, pending_valid;
  Tensor state_values, state_present;
};
// Both tensors of each pair must be supplied, or neither. Undefined pairs mean
// disconnected roots. Connected zero and disconnected poison remain distinct.
struct ResidentCotangents {
  ResidentToken token;
  Tensor outputs, outputs_connected, pending, pending_connected, final, final_connected;
};
struct ResidentBoundaryGradient {
  ResidentToken token;
  // Physical [sample,node,time,kind,source,position], including incoming pending.
  // valid selects actual incoming leaves, connected selects non-None gradients.
  Tensor coordinates, values, valid, connected;
};
struct ResidentGradients {
  std::vector<std::string> names;
  std::vector<std::vector<std::string>> aliases;
  std::vector<Index> offsets;
  Tensor values, connected, initial, initial_connected;
  std::vector<ResidentBoundaryGradient> boundaries;
};
struct ResidentOptimizerState {
  Tensor values, first, second, maximum, steps, corrections;
};
// An explicit CPU export at a detached complete cut. All fields belong to the
// consumer; it may serialize them. No live tape, RNG or data cursor is implied.
struct ResidentTrainingCheckpoint {
  Index schema=1, generation=0, next_token=0;
  Continuation continuation;
  std::map<std::string,Tensor> parameters;
  std::vector<std::vector<std::string>> aliases;
  std::vector<std::string> trainable;
  ResidentOptimizerKind optimizer=ResidentOptimizerKind::sgd;
  std::vector<OptimizerGroup> groups;
  std::vector<Index> offsets;
  ResidentOptimizerState state;
};
struct ResidentStep { bool applied=false; int refusal_code=0; Index generation=0; };

// Explicit first-order VJP API, separate from eager/autograd and inference.
// All methods require no-grad; a consumer computes loss/head cotangents outside
// this owner. Single-NPU FP32 HARD, sum/broadcast, identity/EMA/Add state and
// identity/tanh/LH/SwiGLU Full. Other adjoints are rejected before the first advance.
class ResidentTrainingSession {
 public:
  ResidentTrainingSession(Graph,Model,const Continuation&,at::Device,
      ResidentOptimizerKind kind=ResidentOptimizerKind::sgd,std::vector<OptimizerGroup> groups={},ResidentTrainingLimits limits={});
  ResidentTrainingSession(Graph,Model,const ResidentTrainingCheckpoint&,at::Device,ResidentTrainingLimits={});
  ~ResidentTrainingSession();
  ResidentTrainingSession(const ResidentTrainingSession&)=delete;
  ResidentTrainingSession& operator=(const ResidentTrainingSession&)=delete;
  ResidentTrainingWindow advance(const std::vector<External>&,Index stop,Index sealed_until);
  ResidentGradients backward(const std::vector<ResidentCotangents>&); // All live windows, in forward order.
  ResidentStep step(); // Consume gradients; explicit detach at the update boundary.
  void detach(); // Discard outstanding tapes/gradients, preserve numerical continuation.
  ResidentTrainingCheckpoint checkpoint() const; // Refuses outstanding tapes/gradients.
  Result result() const; // Explicit diagnostics; never an input to device progression.
  Index cut() const;
  Index generation() const;
  Index retained_windows() const;
  void close();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace tide
