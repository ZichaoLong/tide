#include "dispatch.h"
#include <tide/autograd.h>
#include <tide/counters.h>
#include <algorithm>
#include <stdexcept>

namespace accelerator_scale {
Tensor rank_candidates(const Tensor& scores, const Tensor& selected, const Tensor& affected) {
  // Candidates arrive in ascending node order. Stable least-to-most-significant
  // passes implement (selected asc, affected desc, score desc, node asc).
  at::NoGradGuard guard;
  if (scores.dim()!=2 || selected.sizes()!=scores.sizes() || affected.sizes()!=scores.sizes()
      || selected.scalar_type()!=at::kLong || affected.scalar_type()!=at::kLong
      || selected.device()!=scores.device() || affected.device()!=scores.device()
      || !at::isfinite(scores).all().item<bool>())
    throw std::invalid_argument("invalid tensor ranking inputs");
  auto order=at::argsort(scores, true, -1, true);
  order=order.gather(1,at::argsort(affected.gather(1,order),true,-1,true));
  return order.gather(1,at::argsort(selected.gather(1,order),true,-1,false));
}
namespace {
Index count(const History& h,const char* field,Index node) {
  const auto& counts=h.node_maps.at(field);auto it=counts.find(node);
  return it==counts.end()?0:it->second;
}
}
void tensor_select(const Graph& graph,const Model& model,const Placement& placement,
                   std::vector<Event>& events,const std::vector<RegionTask*>& tasks) {
  if(tasks.empty())return;
  const auto shard=placement.node_device[region_layout(graph,tasks.front()->owner.second).members.front()];
  const auto device=placement.devices.at(shard);
  // Exact candidate-size buckets avoid padded scores or approximate tie keys.
  std::map<size_t,std::vector<RegionTask*>> buckets;
  for(auto* task:tasks)buckets[task->ids->size()].push_back(task);
  for(const auto& [width,batch]:buckets) {
    std::vector<History> old; std::vector<std::vector<Tensor>> descriptors;
    std::vector<Index> selected,affected;
    for(auto* task:batch) {
      const auto layout=region_layout(graph,task->owner.second);
      const auto& w=model.regions.at(task->owner.second);
      const auto time=events.at(task->ids->front()).time;
      old.push_back(task->old?*task->old:w.kernel->initial(w,layout,model.nodes[0].bias));
      validate_history(old.back(),layout,model.nodes[0].bias,time);
      w.kernel->validate_history(old.back(),layout);
      descriptors.emplace_back();Index previous=-1;
      for(auto id:*task->ids) {
        const auto& e=events.at(id);
        if(e.node<=previous || e.batch!=task->owner.first || e.time!=time
            || graph.nodes[e.node].region!=task->owner.second)
          throw std::invalid_argument("invalid tensor selection candidate order");
        previous=e.node;descriptors.back().push_back(e.descriptor);
        const bool lh=layout.spec.selector=="lh-count-affect-v1";
        selected.push_back(layout.spec.count_priority?count(old.back(),"selected",e.node):0);
        affected.push_back(lh?count(old.back(),"affected",e.node):0);
      }
    }
    Tensor order;
    {
      at::NoGradGuard guard;Transfer copy(device);
      for(const auto& row:descriptors)for(const auto& d:row)copy.add(d);
      copy.execute();std::vector<Tensor> rows;
      for(const auto& row:descriptors) {
        std::vector<Tensor> values;for(const auto& d:row)values.push_back(copy.get(d));
        rows.push_back(at::stack(values));
      }
      auto shape=std::vector<Index>{static_cast<Index>(batch.size()),static_cast<Index>(width)};
      auto opts=at::TensorOptions().dtype(at::kLong);
      auto a=at::from_blob(selected.data(),shape,opts).clone().to(device);
      auto b=at::from_blob(affected.data(),shape,opts).clone().to(device);
      order=rank_candidates(at::stack(rows),a,b).to(at::kCPU).contiguous();
      if(!device.is_cpu()) {
        transfers.metadata_host_to_device+=2*selected.size()*sizeof(Index);
        transfers.metadata_device_to_host+=order.numel()*sizeof(Index);
      }
    }
    auto indices=order.accessor<Index,2>();
    for(size_t row=0;row<batch.size();++row) {
      auto& task=*batch[row];auto& result=task.selection;
      const auto layout=region_layout(graph,task.owner.second);
      result.history=std::move(old[row]);result.history.last_time=events.at(task.ids->front()).time;
      for(Index k=0;k<std::min<Index>(layout.spec.budget,width);++k)
        result.active.insert(events.at(task.ids->at(indices[row][k])).node);
      // Keep each differentiable softmax independent: batching its autograd
      // graph would turn disconnected roots in other regions into connected zero.
      const auto control_device=placement.scoring.control_device=="cpu"?at::Device(at::kCPU):device;
      Transfer copy(control_device);for(const auto& d:descriptors[row])copy.add(d);copy.execute();
      std::vector<Tensor> values;for(const auto& d:descriptors[row])values.push_back(copy.get(d));
      auto controls=at::softmax(at::stack(values),0).to(at::kFloat);
      if(!at::isfinite(controls).all().item<bool>())throw std::runtime_error("nonfinite tensor controls");
      for(size_t i=0;i<width;++i) {
        const auto node=events.at(task.ids->at(i)).node;
        result.controls.emplace(node,controls[i]);
        if(layout.spec.selector=="lh-count-affect-v1")
          result.history.node_maps["affected"][node]=increment(count(result.history,"affected",node));
        if(result.active.count(node))
          result.history.node_maps["selected"][node]=increment(count(result.history,"selected",node));
      }
    }
  }
}
}  // namespace accelerator_scale
