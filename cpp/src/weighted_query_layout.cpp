#include "weighted_private_rows.h"
#include "tide/operator_profile.h"
#include <ATen/record_function.h>
#include "weighted_query_layout.h"
#include "region_metadata.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
namespace tide::weighted::query {
std::vector<std::vector<Index>> snapshot_chunks(const States& states,Index width,Index budget){
  std::map<Index,std::vector<Index>> buckets;
  for(size_t i=0;i<states.size();++i){
    Index level=0;for(auto n=states[i].slots.at("weights").numel();n;n>>=1)++level;
    buckets[level].push_back(i);
  }
  std::vector<std::vector<Index>> result;
  for(const auto& [level,ids]:buckets){
    Index length=1;for(auto i:ids)length=std::max<Index>(length,states[i].slots.at("weights").numel());
    const auto bytes=length*(6*width*states[ids[0]].value.element_size()+64);
    const auto count=std::max<Index>(1,budget/std::max<Index>(1,bytes));
    for(size_t start=0;start<ids.size();start+=count)
      result.emplace_back(ids.begin()+start,ids.begin()+std::min<size_t>(ids.size(),start+count));
  }
  return result;
}
SnapshotLayout::SnapshotLayout(States values,const Times& times,Index window):states(std::move(values)){
  std::vector<Index> lengths,offsets;Index total=0;rows=states.size();length=1;
  for(const auto& s:states){auto n=s.slots.at("weights").numel();offsets.push_back(total);lengths.push_back(n);total+=n;length=std::max<Index>(length,n);}
  const auto options=states[0].slots.at("times").options();
  auto cols=at::arange(length,options).unsqueeze(0),present=cols<at::tensor(lengths,options).unsqueeze(1);
  indices=at::where(present,at::tensor(offsets,options).unsqueeze(1)+cols,at::full({rows,length},total,options));
  auto delta=at::tensor(times,options).unsqueeze(1)-slot("times");
  visible=present&(delta>=0);if(window)visible=visible&(delta<window);age=delta.clamp_min(0);
  work::packing::bucket("kv_snapshot_layout",length,rows);
}
Tensor SnapshotLayout::slot(const std::string& name)const{
  RECORD_FUNCTION("tide::stage::snapshot_pack", std::vector<c10::IValue>());
  op_profile::Scope stage_profile(op_profile::KvGather);
  std::vector<Tensor> values;for(const auto& s:states)values.push_back(s.slots.at(name));
  auto zero_shape=values[0].sizes().vec();zero_shape[0]=1;
  auto joined=private_rows::join(values);
  auto padded=at::cat({joined,at::zeros(zero_shape,values[0].options())});
  auto shape=values[0].sizes().vec();shape.erase(shape.begin());shape.insert(shape.begin(),{rows,length});
  return padded.index_select(0,indices.flatten()).reshape(shape);
}
Tensor SnapshotLayout::weights()const{auto w=slot("weights");return at::where(visible,w,at::zeros_like(w));}
Tensor SnapshotLayout::visible_slot(const std::string& name)const{
  auto value=slot(name);std::vector<Index> shape(value.dim(),1);shape[0]=rows;shape[1]=length;
  return at::where(visible.reshape(shape),value,at::zeros_like(value));
}
std::vector<std::string> packed_branches(const States& states,const Times& times,const KVConfig& c,bool is_full){
  at::NoGradGuard guard;std::vector<Tensor> flags;std::vector<Index> order;
  for(const auto& ids:snapshot_chunks(states,c.width)){
    States s;Times t;for(auto i:ids){s.push_back(states[i]);t.push_back(times[i]);}
    SnapshotLayout layout(std::move(s),t,c.window);auto writes=layout.weights();
    std::vector<Tensor> columns{(writes>0).any(-1)};
    if(!is_full)columns.push_back((writes*detail::passive_decay(layout.age,c.beta_short,writes)>0).any(-1));
    auto rows=at::stack(columns,-1).flatten().unbind();flags.insert(flags.end(),rows.begin(),rows.end());
    order.insert(order.end(),ids.begin(),ids.end());
  }
  const auto values=region_metadata::flags(flags,true);size_t n=is_full?1:2;
  std::vector<std::string> result(states.size());
  for(size_t j=0;j<order.size();++j){result[order[j]]=":long="+std::to_string(values[j*n]);
    if(!is_full)result[order[j]]+=":short="+std::to_string(values[j*n+1]);}
  return result;
}
}
