#include "tide/region.h"
#include "tide/ownership.h"
#include "region_metadata.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <optional>
#include <stdexcept>

namespace tide {
namespace {
void check_tensor(const Tensor& value,const Tensor& ref,const char* message,std::optional<bool> finite={}){
  if(!value.defined()||value.device()!=ref.device()||value.scalar_type()!=ref.scalar_type()
    ||!(finite?*finite:at::isfinite(value).all().item<bool>()))throw std::invalid_argument(message);
}
// Owning boundary keeps request references stable through one numeric batch.
struct Frame {
  Owner owner;Index time;RegionLayout layout;Tensor reference;History old;
  std::vector<Candidate> candidates;std::set<Index> nodes;
  Frame(const Graph& g,const Model& m,const History* previous,const std::vector<Event>& events,const std::vector<size_t>& ids)
    :owner{events.at(ids.at(0)).batch,g.nodes[events.at(ids.at(0)).node].region},time(events[ids[0]].time),
     layout(region_layout(g,owner.second)),reference(region_reference(g,m,owner.second)){
    Index last=-1;
    for(auto i:ids){const auto& e=events[i];
      if(e.batch!=owner.first||e.time!=time||e.node<=last||g.nodes[e.node].region!=owner.second)
        throw std::invalid_argument("invalid region candidate domain/order");
      candidates.push_back({e.node,e.descriptor,e.descriptor_fields});nodes.insert(e.node);last=e.node;
    }
    const auto& w=m.regions[owner.second];
    if(previous)old=*previous;
    else{
      old=w.kernel->initial(w,layout,m.nodes[0].bias.to(reference.device()));
      validate_history(old,layout,reference,time-1);w.kernel->validate_history(old,layout);
    }
  }
  RegionInput request() const{return {old,time,candidates,layout,reference.options()};}
  Selection validate(const Model& m,Selection result,const std::map<const void*,bool>* finite=nullptr) const{
    if(static_cast<Index>(result.active.size())>layout.spec.budget
      ||!std::includes(nodes.begin(),nodes.end(),result.active.begin(),result.active.end()))
      throw std::invalid_argument("region selector returned invalid active subset/capacity");
    if(result.controls.size()!=nodes.size())throw std::invalid_argument("region selector returned invalid control domain");
    for(const auto& [node,control]:result.controls){
      if(!nodes.count(node))throw std::invalid_argument("region selector returned invalid control domain");
      std::optional<bool> known;
      if(finite&&control.defined()){
        const auto found=finite->find(control.unsafeGetTensorImpl());if(found!=finite->end())known=found->second;
      }
      check_tensor(control,reference,"region selector returned incompatible or nonfinite control",known);
    }
    validate_history(result.history,layout,reference,time);m.regions[owner.second].kernel->validate_history(result.history,layout);
    for(auto& [node,control]:result.controls)control=control.to(m.nodes[node].bias.device());return result;
  }
};
}
std::vector<Selection> RegionKernel::batch_grad(const RegionWeights&,const std::vector<RegionInput>&) const{
  throw std::invalid_argument("region program does not declare batched VJP");
}
void validate_history(const History& h,const RegionLayout& layout,const Tensor& ref,Index time){
  if(h.last_time < -1||h.last_time>time)throw std::invalid_argument("invalid region history clock");
  for(const auto& [name,value]:h.scalars)if(name.empty())throw std::invalid_argument("invalid region history field names");
  for(const auto& [name,counts]:h.node_maps){
    if(name.empty())throw std::invalid_argument("invalid region history field names");
    for(const auto& [node,count]:counts)layout.slot(node);
  }
  for(const auto& [name,value]:h.tensors){
    if(name.empty())throw std::invalid_argument("invalid region history field names");
    check_tensor(value,ref,"incompatible or nonfinite region history tensor");
  }
}
Selection evaluate_selection(const Graph& g,const Model& m,const History* previous,const std::vector<Event>& events,const std::vector<size_t>& ids){
  if(ids.empty())throw std::invalid_argument("empty region evaluation");
  Frame f(g,m,previous,events,ids);const auto& w=m.regions[f.owner.second];return f.validate(m,w.kernel->step(w,f.request()));
}
void commit_selection(Continuation& q,Owner owner,Selection result,std::vector<Event>& events,const std::vector<size_t>& ids,bool trace){
  q.history[owner]=std::move(result.history);
  for(auto i:ids){auto& e=events[i];e.active=result.active.count(e.node);e.control=std::move(result.controls.at(e.node));if(trace)e.history=q.history.at(owner);}
}
void select_events(const Graph& g,const Model& m,Continuation& q,std::vector<Event>& events,const std::vector<size_t>& ids,bool trace){
  if(ids.empty())return;
  const auto& first=events[ids[0]];const Owner owner{first.batch,g.nodes[first.node].region};const auto it=q.history.find(owner);
  auto result=evaluate_selection(g,m,it==q.history.end()?nullptr:&it->second,events,ids);
  commit_selection(q,owner,std::move(result),events,ids,trace);
}
void select_event_batches(const Graph& g,const Model& m,Continuation& q,std::vector<Event>& events,
    const std::map<Owner,std::vector<size_t>>& owners,bool trace,std::map<std::string,Index>& stats){
  std::map<Index,std::vector<Owner>> groups;
  for(const auto& [owner,ids]:owners)groups[owner.second].push_back(owner);
  for(const auto& [region,used]:groups){
    const auto& w=m.regions[region];
    if(!w.kernel->joint_batch()||(at::GradMode::is_enabled()&&!w.kernel->batched_autograd())){
      for(auto owner:used){select_events(g,m,q,events,owners.at(owner),trace);++stats["region_scalar_calls"];}
      continue;
    }
    std::vector<Frame> frames;frames.reserve(used.size());
    for(auto owner:used){const auto old=q.history.find(owner);frames.emplace_back(g,m,old==q.history.end()?nullptr:&old->second,events,owners.at(owner));}
    std::vector<RegionInput> requests;for(const auto& f:frames)requests.push_back(f.request());
    auto results=at::GradMode::is_enabled()?w.kernel->batch_grad(w,requests):w.kernel->batch(w,requests);
    if(results.size()!=frames.size())throw std::invalid_argument("region batch returned wrong frame count");
    std::map<const void*,bool> finite;
    const bool grouped=frames.size()>1&&!frames[0].reference.device().is_cpu();
    if(grouped){
      std::vector<Tensor> controls;
      for(size_t i=0;i<frames.size();++i)for(const auto& [node,c]:results[i].controls)
        if(c.defined()&&c.device()==frames[i].reference.device()&&c.scalar_type()==frames[i].reference.scalar_type())controls.push_back(c);
      const auto flags=region_metadata::flags(controls,false);
      for(size_t i=0;i<controls.size();++i)finite[controls[i].unsafeGetTensorImpl()]=flags[i];
    }
    // Validate every actual result before committing any owner in this batch.
    for(size_t i=0;i<frames.size();++i)results[i]=frames[i].validate(m,std::move(results[i]),grouped?&finite:nullptr);
    for(size_t i=0;i<frames.size();++i)commit_selection(q,used[i],std::move(results[i]),events,owners.at(used[i]),trace);
    ++stats["region_batch_calls"];stats["max_region_batch"]=std::max<Index>(stats["max_region_batch"],used.size());
    if(at::GradMode::is_enabled())stats["batched_region_frames"]+=used.size();
  }
}
}
