#pragma once
#include "../../cpp/scale/scale.h"
#include "scoring.h"
#include "profiling.h"
#include <tide/kernel.h>
#include <tide/full.h>
#include <atomic>
#include <mutex>

namespace accelerator_scale {
using namespace tide;
struct Counters {
  std::atomic<int64_t> host_to_device{0}, device_to_host{0}, copies{0};
  std::atomic<int64_t> device_to_device{0}, local_messages{0}, remote_messages{0};
  std::atomic<int64_t> metadata_host_to_device{0}, metadata_device_to_host{0};
  void reset();
  std::map<std::string, double> metrics() const;
};
extern Counters transfers;

// Coalesce fields only without autograd; keep independent gradient roots apart.
// Only explicit copies are counted; scalar extraction inside selectors is not.
class Transfer {
 public:
  explicit Transfer(at::Device destination) : destination_(destination) {}
  void add(const Tensor&);
  void add(const State&);
  void add(const ContentView&);
  void execute();
  Tensor get(const Tensor&) const;
  State get(const State&) const;
 private:
  at::Device destination_;
  std::map<const c10::TensorImpl*, size_t> index_;
  std::vector<Tensor> source_, result_;
};
struct ContentStorage {
  Tensor value;
  std::vector<SourceInput> sources;
  std::vector<SlotValue> contributions;
  std::shared_ptr<SourceBatch> batch;
  Index row;
  ContentStorage(const ContentView&, const Transfer&);
  ContentView view() const;
};
struct DeviceNode {
  NodeWeights weights;
  // One stream per device is used. Hold this lock across one complete delegated
  // operation, including its blocking return transfer, to bound concurrent arenas.
  std::shared_ptr<std::mutex> mutex;
};
std::shared_ptr<const StateKernel> state_adapter(std::shared_ptr<DeviceNode>);
std::shared_ptr<const FullKernel> full_adapter(std::shared_ptr<DeviceNode>);
struct Placement {
  std::vector<at::Device> devices;
  std::vector<Index> node_device;
  std::vector<int64_t> parameter_bytes;
  std::string policy;
  Index cut_edges = 0, edges = 0;
  int64_t node_load_limit = 0;
  bool resident = false;
  Scoring scoring;
  ProfileConfig profile;
  std::string ranking_device = "cpu", event_device = "cpu";
};
struct Partition { std::vector<Index> shards; int64_t limit; };
Partition partition(const Graph&, const std::vector<int64_t>&, Index, const std::string&);
Placement place(pdg_scale::Fixture&, at::Device first, Index count, const std::string& policy, bool resident);
void synchronize(const Placement&);
void reset_memory(const Placement&);
void finalize();
std::vector<Tensor> initialize_devices(at::Device first, Index count);
std::map<std::string, double> memory(const Placement&);
Tensor host(const Tensor&);
Tensor embed(const Tensor&, const Tensor& ids, bool host_result = true);
Tensor project(const Tensor& hidden, const Tensor& weight, bool host_result = true);
void check(const pdg_scale::Config&, const pdg_scale::Topology&, at::Device, Index devices, const std::string& policy, bool resident, bool conditioned, const Scoring&, bool reference_fp64, const std::string& ranking_device, const std::string& event_device);
}  // namespace accelerator_scale
