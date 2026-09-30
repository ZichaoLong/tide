#include "retained_fixture.h"
#include "tide/stream.h"
#include "tide/parameters.h"
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <limits>

namespace tide::device_online::test {
Fixture retained_fixture(int shape,int variant,int64_t width) {
  auto f=graph_vjp_fixture(shape,variant,width);
  f.model.nodes[2].weight=f.model.nodes[0].weight;f.model.nodes[2].bias=f.model.nodes[0].decay;f.model.nodes[3].read=f.model.nodes[0].bias;
  f.model.input_scale[1]=f.model.agg_scale[0];f.model.output_scale[2]=f.model.edge_scale[1];return f;
}
std::vector<int64_t> retained_stops(int64_t start){return {start+3,start+7,start+11,start+11};}
namespace {
bool output_root(int mode){return mode==1||mode==4;}
bool final_root(int window,int mode){return mode==4||window==3&&(mode==2||mode==5);}
bool pending_root(int window,int mode){return mode==4||window==3&&mode==3;}
}
GraphCotangents retained_roots(const ReverseTape& t,int window,int mode) {
  auto opts=t.fiber_values.options();const float nan=std::numeric_limits<float>::quiet_NaN();
  GraphCotangents r{at::full_like(t.outputs.values,nan),at::zeros_like(t.outputs.valid),at::full_like(t.pending.values,nan),at::zeros_like(t.pending.valid),
    at::full({t.state.samples,int64_t(t.graph->nodes.size()),t.full.width},nan,opts),at::zeros({t.state.samples,int64_t(t.graph->nodes.size())},opts.dtype(at::kBool))};
  if(output_root(mode)){r.outputs.fill_(.0625f);r.outputs_connected.copy_(t.outputs.valid);}
  if(pending_root(window,mode)){r.pending.fill_(.015625f);r.pending_connected.copy_(t.pending.valid);}
  if(final_root(window,mode)){r.final.fill_(mode==5?0.f:.03125f);r.final_connected.fill_(true);}
  return r;
}
RetainedReference retained_reference(Fixture f,int mode,at::ScalarType dtype) {
  at::AutoGradMode enabled(true);RetainedReference result;std::vector<Tensor> leaves,terms;std::vector<std::string> names;
  std::map<const void*,Tensor> copies;
  auto copy=[&](Tensor& x){auto key=x.unsafeGetTensorImpl();auto it=copies.find(key);
    if(it==copies.end())it=copies.emplace(key,x.detach().to(dtype).clone().set_requires_grad(true)).first;x=it->second;};
  for(auto& w:f.model.nodes){copy(w.decay);copy(w.weight);copy(w.bias);copy(w.read);for(auto& [_,x]:w.extra)copy(x);}
  for(auto* group:{&f.model.input_scale,&f.model.agg_scale,&f.model.edge_scale,&f.model.output_scale})for(auto& x:*group)copy(x);
  for(auto& owner:f.model.parameters(false).owners()){leaves.push_back(owner.value);names.push_back(owner.canonical);}
  auto leaf=[&](const std::string& name,Tensor& x){x=x.detach().to(dtype).clone().set_requires_grad(true);leaves.push_back(x);names.push_back(name);};
  for(auto& [owner,s]:f.initial.states)leaf("state/"+std::to_string(owner.first)+"/"+std::to_string(owner.second),s.value);
  for(auto& a:f.initial.pending)leaf(boundary_name(a),a.value);
  for(auto& x:f.input)leaf(boundary_name({x.batch,f.graph.inputs[x.port],x.time,0,x.port,x.position,x.value}),x.value);
  auto q=f.initial;int window=0;
  for(auto stop:retained_stops(q.cut)) {
    std::vector<External> input;for(const auto& x:f.input)if(x.time>=q.cut&&x.time<stop)input.push_back(x);
    Streaming cpu(f.graph,f.model,{});auto r=cpu.run(q,input,stop,stop);
    if(output_root(mode))for(const auto& o:r.outputs)terms.push_back(o.value.sum()*.0625);
    if(pending_root(window,mode))for(const auto& a:r.continuation.pending)terms.push_back(a.value.sum()*.015625);
    if(final_root(window,mode))for(const auto& [_,s]:r.continuation.states)terms.push_back(s.value.sum()*(mode==5?0.:.03125));
    q=r.continuation;result.windows.push_back(std::move(r));++window;
  }
  std::vector<Tensor> gradients(leaves.size());if(!terms.empty())gradients=torch::autograd::grad({at::stack(terms).sum()},leaves,{},false,false,true);
  for(size_t i=0;i<leaves.size();++i)result.gradients.emplace(names[i],gradients[i]);return result;
}
} // namespace tide::device_online::test
