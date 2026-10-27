#include "weighted_stage_check.h"
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
namespace tide::device_online::stage_check {
namespace {
void require(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
void close(const at::Tensor& actual,const at::Tensor& expected,const std::string& label){
  require(actual.sizes()==expected.sizes(),label+": shape differs");
  if(!at::allclose(actual.cpu(),expected,3e-5,3e-6)){std::cerr<<label<<" actual "<<actual.cpu()<<" expected "<<expected<<'\n';throw std::runtime_error(label+": value differs");}
}
void gradient(const at::Tensor& actual,bool connected,const at::Tensor& expected,const std::string& label){
  require(connected==expected.defined(),label+": None/zero differs actual="+std::to_string(connected));
  if(connected)close(actual,expected,label);else require(actual.eq(0).all().item<bool>(),label+": absent gradient nonzero");
}
const at::Tensor& field(const State& s,const std::string& name){return name=="value"?s.value:s.slots.at(name);}
}
void verify(const Fixture& f,const Result& reference,const ReadyBatch& ready,const WeightedFlowNodes& nodes,
    const std::vector<WeightedNodeRoots>& roots,const std::map<int64_t,WeightedRowAdjoint>& controls,
    const std::vector<WeightedNodeGradient>& g,const ParameterVjp& parameters,const std::string& kind){
  at::AutoGradMode enabled(true);std::vector<at::Tensor> output,bars,leaves;
  auto add=[&](const at::Tensor& y,const at::Tensor& bar){if(y.requires_grad()){output.push_back(y);bars.push_back(bar.cpu().reshape(y.sizes()));}};
  for(const auto& e:reference.trace){const auto& n=*nodes[e.node];const auto& root=roots[e.node];const auto b=e.batch;
    close(n.aggregate->output[b],e.content,"content");close(n.control[b],e.control,"control");
    require(n.active[b].cpu().item<bool>()==e.active,"stage active support differs");
    for(const auto& [name,x]:e.descriptor_fields)close(n.describe->fields.at(name)[b].reshape(x.sizes()),x,"Describe."+name);
    if(e.active)close(n.full[b],e.full.narrow(0,1,e.full.numel()-1),"Full");
    if(root.content.connected[b].item<bool>())add(e.content,root.content.value[b]);
    // A selected Full root already contains c. A separately requested control
    // of an inactive candidate is still a real root (e.g. zero-base sending's
    // declared right derivative); do not silently drop it in the combined case.
    if(controls.at(e.node).connected[b].item<bool>()&&!(e.active&&root.full.connected[b].item<bool>()))
      add(e.control,controls.at(e.node).value[b]);
    if(e.active&&root.full.connected[b].item<bool>())add(e.full,at::cat({controls.at(e.node).value[b],root.full.value[b]}));
    for(const auto& [name,part]:root.describe)if(part.connected[b].item<bool>())add(e.descriptor_fields.at(name),part.value[b]);
  }
  for(size_t v=0;v<nodes.size();++v){const auto& n=*nodes[v];at::NoGradGuard ng;
    auto decoded=decode_weighted_raw_state(n.layout,copy_weighted_raw_state(n.proposal,at::kCPU));
    for(int64_t b=0;b<rows;++b){const auto& s=reference.continuation.states.at({b,int64_t(v)});
      require(s.last_time==decoded[b].last_time&&s.observations==decoded[b].observations,"raw clock/count differs");
      for(const auto& [name,x]:s.slots)if(x.scalar_type()==at::kLong)require(at::equal(decoded[b].slots.at(name),x),"raw metadata differs");
      for(size_t j=0;j<n.layout.fields().size();++j){const auto& part=n.layout.fields()[j];const auto& x=field(s,part.name);
        close(field(decoded[b],part.name),x,"raw."+part.name);
        if(roots[v].proposal.connected[b][j].item<bool>())add(x,roots[v].proposal.value[b].narrow(0,part.offset,x.numel()));
      }
    }
  }
  auto registry=f.model.parameters();for(const auto& owner:registry.owners())leaves.push_back(owner.value);
  for(const auto& x:f.inputs)leaves.push_back(x.value);
  for(const auto& [owner,state]:f.initial.states){leaves.push_back(state.value);for(const auto& [name,x]:state.slots)if(x.scalar_type()==at::kFloat)leaves.push_back(x);}
  auto expected=output.empty()?std::vector<at::Tensor>(leaves.size()):torch::autograd::grad(output,leaves,bars,true,false,true);
  size_t index=0;auto parameter_values=parameters.values.cpu(),parameter_flags=parameters.connected.cpu();
  for(size_t j=0;j<parameters.owners.size();++j){const auto& owner=parameters.owners[j];const auto offset=parameters.offsets[j];
    auto actual=offset<0?at::zeros_like(owner.value):parameter_values.narrow(0,offset,owner.value.numel()).reshape(owner.value.sizes());
    gradient(actual,parameter_flags[j].item<bool>(),expected.at(index++),owner.canonical);}
  auto meta=ready.atoms.coordinates.cpu();
  for(const auto& x:f.inputs){const auto v=f.graph.inputs[x.port];const auto& n=*nodes[v];auto ids=n.input.indices.cpu();bool found=false;
    for(int64_t j=0;j<n.input.lengths[x.batch].cpu().item<int64_t>();++j){auto atom=meta[ids[x.batch][j].item<int64_t>()];
      if(atom[3].item<int64_t>()==0&&atom[4].item<int64_t>()==x.port){
        require(!found,"duplicate physical input in stage");found=true;gradient(g[v].payload[x.batch][j],g[v].connected[x.batch][j].cpu().item<bool>(),expected.at(index),"input "+std::to_string(index));}}
    require(found,"missing physical input in stage");++index;
  }
  for(const auto& [owner,state]:f.initial.states){const auto b=owner.first,v=owner.second;const auto& n=*nodes[v];auto value=g[v].old.value.cpu(),flags=g[v].old.connected.cpu();
    auto one=[&](const std::string& name,const at::Tensor& original){const auto& part=n.layout.field(name);size_t j=0;while(n.layout.fields()[j].name!=name)++j;
      gradient(value[b].narrow(0,part.offset,original.numel()).reshape(original.sizes()),flags[b][j].item<bool>(),expected.at(index++),"old."+name+"/"+std::to_string(b)+"/"+std::to_string(v));};
    one("value",state.value);for(const auto& [name,x]:state.slots)if(x.scalar_type()==at::kFloat)one(name,x);
  }
  require(index==expected.size(),"stage gradient roots not completely checked");
}
}
