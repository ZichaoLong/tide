// Fixed 0770593 KV §1.2: actual positive writes on node×1×R tensors.
#include "weighted_tensor_algebra.h"
#include <ATen/core/grad_mode.h>

namespace tide::weighted::columns::tensor {
namespace {
Packed pack_writes(const std::vector<WriteRequest>& requests){
  at::NoGradGuard guard;Packed p;p.nodes=requests.size();p.batch=1;
  Index total=0;std::vector<Index> lengths;
  for(const auto& r:requests){p.queries=std::max(p.queries,static_cast<Index>(r.rows->size()));
    lengths.push_back(r.content.size(0));total+=r.content.size(0);}
  std::vector<Index> mapping(p.nodes*p.queries,total),inverse(total,p.nodes*p.queries);
  Index base=0;const auto device=requests[0].content.device();
  for(Index g=0;g<p.nodes;++g){const auto& rows=*requests[g].rows;std::vector<Index> publish;
    for(size_t j=0;j<rows.size();++j){mapping[g*p.queries+j]=base+rows[j];inverse[base+rows[j]]=g*p.queries+j;publish.push_back(g*p.queries+j);}
    p.publish.push_back(indices(publish,device));base+=lengths[g];
  }
  auto index=indices(mapping,device),restore=indices(inverse,device);
  auto data=[&](const std::string& name,auto value){Role role;role.name=name;role.lengths=lengths;role.restore=restore;
    for(const auto& r:requests)role.originals.push_back(value(r));
    auto tensors=role.originals;auto shape=tensors[0].sizes().vec();shape[0]=1;tensors.push_back(at::zeros(shape,tensors[0].options()));
    shape[0]=p.queries;shape.insert(shape.begin(),{p.nodes,1});
    role.value=at::cat(tensors).index_select(0,index).reshape(shape);p.values[name]=role.value;p.roles.push_back(std::move(role));
  };
  data("$u",[](const WriteRequest& r){return r.content;});data("$mass",[](const WriteRequest& r){return r.mass;});
  for(const auto* name:{"kernel.base","kernel.adapter_v","kernel.adapter_u","kernel.summary"}){Role role;role.name=name;role.parameter=true;role.shared=true;
    for(const auto& r:requests)role.originals.push_back(detail::weight(*r.owner,name));
    for(const auto& t:role.originals)role.shared&=t.is_same(role.originals[0]);
    if(role.shared){role.value=role.originals[0].detach();p.shared.insert(name);}else role.value=at::stack(role.originals);
    p.values[name]=role.value;p.roles.push_back(std::move(role));
  }
  work::packing::bucket("kv_tensor_writes",p.queries,p.nodes);return p;
}
TensorFields writes(const Packed& p,double epsilon){
  Algebra a{p};auto normalized=normalize(a.at("$u"),epsilon);
  auto kv=a.project(normalized,"kernel.base")+a.project(a.project(normalized,"kernel.adapter_v"),"kernel.adapter_u");
  const Index width=kv.size(-1)/2;
  return {{"keys",kv.slice(-1,0,width)},{"values",kv.slice(-1,width)},
    {"summaries",a.project(normalized,"kernel.summary")},{"weights",a.at("$mass").clamp_max(1)}};
}
}
std::vector<TensorFields> prepare(const std::vector<WriteRequest>& requests,double epsilon){
  detail::require(!requests.empty(),"KV tensor writes require a positive-write group");
  return evaluate(pack_writes(requests),[&](const Packed& p){return writes(p,epsilon);},"private/kv.tensor.writes");
}
}
