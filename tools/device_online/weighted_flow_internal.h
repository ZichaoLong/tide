#pragma once
#include "weighted_flow.h"
#include "weighted_flow_parameters.h"
#include "weighted_kv_sum.h"
#include "ready_batch.h"
#include "task_services.h"
#include "queue_transaction.h"
#include "broadcast_router.h"
#include "tide/ops.h"
#include <optional>
namespace tide::device_online {
int64_t weighted_flow_preflight(const Graph&,const Model&,int64_t samples,const WeightedFlowLimits&);
struct WeightedNodeInput {at::Tensor payload,owners,lengths,times,present,bank,indices;};
WeightedNodeInput append_weighted_node_input(DeviceProgram&,const ReadyBatch&,const at::Tensor& source_banks,
  int64_t node,int64_t samples,int64_t input_ports,const at::Tensor& error);
struct WeightedRegionInput {WeightedRegionFields fields;at::Tensor ids,times,lengths,inverse;};
WeightedRegionInput append_weighted_region_input(DeviceProgram&,const at::Tensor& descriptors,const at::Tensor& present,
  const at::Tensor& times,const std::vector<int64_t>& nodes,const weighted::RegionalConfig&,const at::Tensor& error);
void append_weighted_owner_adopt(DeviceProgram&,const at::Tensor& target,const at::Tensor& present,
  const at::Tensor& times,const at::Tensor& error,bool history);
void weighted_copy(DeviceProgram&,const at::Tensor& target,const at::Tensor& source);
at::Tensor weighted_columns(DeviceProgram&,const at::Tensor&,int64_t start,int64_t count);
struct WeightedFlowSource {
  at::Tensor classes;
  WeightedSourceParameters values;
  std::vector<std::optional<WeightedSourceAdaptation>> adaptations;
};
WeightedFlowSource bind_weighted_flow_source(DeviceProgram&,WeightedBanks&,const Graph&,int64_t node,
  const weighted::NodeBinding&,const at::Tensor& error,int64_t budget);
struct WeightedFlowNode {
  weighted::NodeBinding binding;WeightedRawLayout layout;WeightedBanks banks;
  WeightedRawState initial,state,old,proposal;
  at::Tensor state_present,control,active,full,content,mass;
  WeightedFlowSource source;WeightedNodeInput input;
  std::optional<WeightedAggregateTape> aggregate;
  std::optional<WeightedKVUpdateTape> kv_update;
  std::optional<WeightedKVSumTape> kv_sum;
  std::optional<WeightedKVQueryTape> kv_query;
  std::optional<WeightedKVFullTape> kv_full;
  std::optional<WeightedRecurrentTape> recurrent_update,recurrent_mem,recurrent_full;
  WeightedRecurrentParameters recurrent_parameters;
  std::optional<WeightedDescribeTape> describe;
  WeightedFlowNode(const Model&,int64_t node,const Continuation&,at::Device,int64_t capacity,int64_t budget);
};
struct WeightedFlowRegion {
  weighted::RegionalConfig config;WeightedBanks banks;std::vector<int64_t> nodes;
  WeightedRegionInput input;std::optional<WeightedRegionTape> selection;at::Tensor history;
  WeightedFlowRegion(const Graph&,const Model&,int64_t,const Continuation&,at::Device);
};
// A bounded window journal, independent of reclaimable queue capacity. The
// current stage is device int64; no host-selected capture sequence is supplied.
class WeightedWindowJournal {
 public:
  WeightedWindowJournal(int64_t stages,int64_t bytes):stages_(stages),remaining_(bytes){}
  void save(DeviceProgram&,const at::Tensor& stage,const std::string&,const at::Tensor&);
  std::map<std::string,at::Tensor> decode(int64_t stages)const;
  int64_t allocated_bytes()const{return allocated_;}
 private:
  int64_t stages_,remaining_,allocated_=0;
  std::map<std::string,at::Tensor> fields_;
};
struct WeightedFlow::Impl {
  Graph graph;Model model;Continuation boundary;at::Device device;WeightedFlowLimits limits;
  std::unique_ptr<TaskServices> services;
  std::unique_ptr<DeviceProgram> program;std::unique_ptr<QueueTransaction> pending;
  std::vector<std::unique_ptr<WeightedFlowNode>> nodes;
  std::vector<std::unique_ptr<WeightedFlowRegion>> regions;
  std::unique_ptr<WeightedWindowJournal> journal;
  at::Tensor error,stop,stages;AtomBatch external;
  bool failed=false,ran=false;int64_t bytes=0,journal_bytes=0;
  Impl(Graph,Model,const Continuation&,at::Device,WeightedFlowLimits);
  void construct();
  int64_t remaining(DeviceProgram&)const;
  void append_nodes(DeviceProgram&,const ReadyBatch&);
  void append_regions(DeviceProgram&);
  ActionBatch append_full(DeviceProgram&);
  void save_stage(DeviceProgram&,const ReadyBatch&,const AtomBatch&);
  Continuation checkpoint()const;
  Result decode()const;
};
}
