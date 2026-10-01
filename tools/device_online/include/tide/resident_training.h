#pragma once
#include "tide/resident.h"
#include "tide/optimizer.h"

namespace tide {
enum class ResidentOptimizerKind { sgd, adamw };
// Empty devices keeps the original single-device owner. Otherwise devices[0]
// must equal the session coordinator. Empty owner maps request generic planning;
// explicit maps index this logical device list, one entry per graph node.
struct ResidentPlacement {
  std::vector<at::Device> devices;
  std::string policy="locality";
  std::vector<Index> full_owners,state_owners;
};
struct ResidentTrainingLimits {
  ResidentLimits forward;
  Index windows=8, retained_bytes=128*1024*1024, backward_bytes=512*1024*1024;
  Index optimizer_bytes=128*1024*1024, program_workspace_bytes=64*1024*1024;
  Index reverse_chunk_rows=16;
  ResidentPlacement placement;
};
struct ResidentToken { uint64_t session=0; Index index=0, generation=0; };
// Owner order is sample-major, followed by the static nodes list. Key/value
// are [owners,capacity,kv_heads,head_width]; lengths and present are [owners].
// Empty-but-present caches may have connected-zero gradients. Same-fiber groups
// additionally expose log_bias [owners,capacity]; event groups leave it undefined.
// Event groups precede fiber groups, each grouped by head geometry.
struct ResidentCacheWindow {
  std::vector<Index> nodes;
  Tensor key,value,lengths,present;
  Tensor log_bias;
};
struct ResidentCacheCotangents {Tensor key,value,key_connected,value_connected,log_bias,log_bias_connected;};
struct ResidentCacheGradient {
  std::vector<Index> nodes;
  Tensor key,value,lengths,key_connected,value_connected;
  Tensor log_bias,log_bias_connected;
};
struct ResidentStateWindow {
  std::vector<Index> nodes;
  Tensor values,present; // [sample,local_node,width] and [sample,local_node].
  std::vector<ResidentCacheWindow> cache; // Global node IDs, owner-local storage.
};
struct ResidentStateCotangents {
  Tensor final,final_connected;
  std::vector<ResidentCacheCotangents> cache;
};
struct ResidentStateGradient {
  std::vector<Index> nodes;
  Tensor initial,initial_connected;
  std::vector<ResidentCacheGradient> cache;
};
struct ResidentParameterGradient {
  std::vector<std::string> names;
  std::vector<std::vector<std::string>> aliases;
  std::vector<Index> offsets;
  Tensor values,connected;
};
struct ResidentTrainingWindow {
  ResidentToken token;
  Index start=0, stop=0;
  ResidentWindow outputs;
  Tensor pending_coordinates, pending_values, pending_valid;
  Tensor state_values, state_present;
  std::vector<ResidentCacheWindow> cache;
  // Sharded sessions expose states here; legacy dense state/cache fields are
  // undefined/empty. No coordinator replica is created for a consumer's loss.
  std::vector<ResidentStateWindow> states;
};
// Both tensors of each pair must be supplied, or neither. Undefined pairs mean
// disconnected roots. Connected zero and disconnected poison remain distinct.
// Values are FP32 and masks are bool, independently of the forward payload dtype.
struct ResidentCotangents {
  ResidentToken token;
  Tensor outputs, outputs_connected, pending, pending_connected, final, final_connected;
  std::vector<ResidentCacheCotangents> cache;
  std::vector<ResidentStateCotangents> states; // Match window.states order.
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
  std::vector<ResidentCacheGradient> initial_cache;
  std::vector<ResidentParameterGradient> parameter_shards;
  std::vector<ResidentStateGradient> initial_shards;
};
struct ResidentOptimizerState {
  // Values and floating slots remain FP32 masters, including FP16 payload runs.
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
  // Legacy v1 records omitted these fields and mean HARD with zeta=1.
  std::string mode="hard";
  double zeta=1.;
};
struct ResidentStep { bool applied=false; int refusal_code=0; Index generation=0; };

// Explicit first-order VJP API, separate from eager/autograd and inference.
// All methods require no-grad; a consumer computes loss/head cotangents outside
// this owner. Explicit single/sharded NPU FP32/FP16 HARD/HST/SOFTP, FP32 adjoints/masters,
// built-in Aggregate/broadcast, identity/EMA/Add/event/fiber-attention state and
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
  ResidentPlacement placement() const; // Resolved static placement.
  void close();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace tide
