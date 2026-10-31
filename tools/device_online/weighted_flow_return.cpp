#include "weighted_flow_internal.h"
#include <algorithm>
namespace tide::device_online {
std::vector<Output> WeightedFlow::Impl::outputs()const{
  const auto count=stages.item<int64_t>();
  if(count<0||count>limits.stages)throw std::logic_error("invalid weighted output journal length");
  if(!count||graph.outputs.empty())return {};
  const auto samples=boundary.batch_size,width=model.width(),rows=count*samples;
  // Bound transient ATen compaction independently of model/journal size. One
  // exceptionally wide payload row is indivisible, as in the eager copy path.
  constexpr int64_t pack_bytes=8*1024*1024;
  const auto row_bytes=3*(width*model.nodes[0].bias.element_size()+4*int64_t(sizeof(Index))+1);
  const auto max_rows=std::max<int64_t>(1,pack_bytes/row_bytes);
  int64_t packed_rows=0;
  std::vector<at::Tensor> coordinates,values,present;
  std::vector<Output> out;
  auto flush=[&]{
    if(values.empty())return;
    auto indices=at::nonzero(at::cat(present,0)).reshape({-1});
    auto coords=at::cat(coordinates,0).index_select(0,indices).to(at::kCPU).contiguous();
    auto payload=at::cat(values,0).index_select(0,indices).to(at::kCPU);
    auto c=coords.accessor<int64_t,2>();
    for(int64_t row=0;row<coords.size(0);++row)out.push_back({c[row][0],c[row][1],c[row][2],payload[row].clone()});
    coordinates.clear();values.clear();present.clear();packed_rows=0;
  };
  for(size_t port=0;port<graph.outputs.size();++port){
    const auto prefix="node."+std::to_string(graph.outputs[port])+".";
    const auto field=[&](const std::string& name){return journal->field(prefix+name).narrow(0,0,count);};
    auto times=field("times").reshape({-1}),payload=field("payload").reshape({-1,width});
    auto observed=field("present").reshape({-1}),active=field("active").reshape({-1});
    for(int64_t start=0;start<rows;){
      const auto size=std::min(rows-start,max_rows-packed_rows);
      auto batches=at::arange(start,start+size,stages.options()).remainder(samples);
      coordinates.push_back(at::stack({batches,times.narrow(0,start,size),at::full_like(batches,port)},-1));
      values.push_back(payload.narrow(0,start,size));
      present.push_back(observed.narrow(0,start,size)&active.narrow(0,start,size));
      start+=size;packed_rows+=size;if(packed_rows==max_rows)flush();
    }
  }
  // Compact actual physical output rows on device; no state, descriptor, fiber
  // or message field is decoded. Masks, rather than payload zeros, own presence.
  flush();
  std::sort(out.begin(),out.end(),[&](const Output& a,const Output& b){
    return std::tie(a.time,a.batch,graph.outputs[a.port],a.port)<std::tie(b.time,b.batch,graph.outputs[b.port],b.port);});
  return out;
}
std::map<std::string,Index> WeightedFlow::Impl::metrics()const{
  // Full capacity contains earlier windows' stale rows. The current device
  // stage mask is mandatory even after an empty or shorter subsequent window.
  auto stage=at::arange(limits.stages,stages.options()).lt(stages).reshape({limits.stages,1});
  auto counts=(journal->field("counts")*stage).sum(0);
  std::vector<at::Tensor> fulls;
  std::vector<int64_t> output_ports(nodes.size(),0);
  for(auto v:graph.outputs)++output_ports[v];
  for(size_t v=0;v<nodes.size();++v){const auto prefix="node."+std::to_string(v)+".";
    fulls.push_back((journal->field(prefix+"present")&journal->field(prefix+"active")&stage).sum(at::kLong));}
  auto per_node=at::stack(fulls);
  auto ports=at::tensor(output_ports,at::kLong).to(device);
  auto messages=(journal->field("message.present")&stage).sum(at::kLong);
  auto outputs=(per_node*ports).sum(at::kLong);
  auto pending_count=pending->atoms().valid.sum(at::kLong);
  auto actual=at::stack({stages.reshape({}),counts[0],counts[1],per_node.sum(at::kLong),
                        messages,outputs,pending_count}).to(at::kCPU).contiguous();
  const auto* c=actual.const_data_ptr<int64_t>();
  auto out=statistics(c[0],c[1],c[2],c[3],c[4]);
  out["stages"]=c[0];out["full_calls"]=c[3];out["outputs"]=c[5];out["pending"]=c[6];
  return out;
}
}
