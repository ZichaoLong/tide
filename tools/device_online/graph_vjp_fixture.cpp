#include "graph_vjp_fixture.h"
#include "full_vjp_fixture.h"
#include "precision_graph_fixture.h"
#include "tide/stream.h"
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <sstream>

namespace tide::device_online::test {
std::string boundary_name(const Atom& a) {
  std::ostringstream key;key<<"boundary/"<<a.batch<<'/'<<a.node<<'/'<<a.time<<'/'<<a.kind<<'/'<<a.source<<'/'<<a.position;return key.str();
}
Fixture graph_vjp_fixture(int shape,int variant,int64_t width) {
  at::NoGradGuard guard;auto f=fixture(shape,variant);
  for(size_t n=0;n<f.graph.nodes.size();++n) {
    auto& node=f.graph.nodes[n];auto& w=f.model.nodes[n];if(!node.identity)node.full=shape==2?"identity":"tanh";
    w.weight.mul_(.125);w.bias.fill_(.03125);w.decay.fill_(-.125);
    if((shape==1||shape==3)&&n==0){node.memory="lh-add-repeat-v1";w.extra["add_retention"]=at::full({},shape==1?.5f:-.5f,at::kFloat);}
    for(Index b=0;b<f.initial.batch_size;++b)if(!f.initial.states.count({b,Index(n)}))
      f.initial.states[{b,Index(n)}]={at::full({3},.015625f,at::kFloat),f.initial.cut-1,0};
  }
  if(shape==3&&variant==1)for(size_t n=0;n<f.graph.nodes.size();++n)if(!f.graph.nodes[n].identity) {
    auto& node=f.graph.nodes[n];node.emit_period=3;
    for(Index j=f.graph.outgoing_ports.offsets[n];j<f.graph.outgoing_ports.offsets[n+1];++j)
      node.emit_phases.push_back((j-f.graph.outgoing_ports.offsets[n])%3==0?0:(j-f.graph.outgoing_ports.offsets[n])%3==1?-2:-1);
  }
  if(!f.model.edge_scale.empty())f.model.edge_scale[0].zero_();
  auto expand=[&](const Tensor& x){return x.repeat({(width+2)/3}).narrow(0,0,width).clone();};
  for(auto& w:f.model.nodes){w.decay=expand(w.decay);w.bias=expand(w.bias);w.read=expand(w.read);w.weight=at::eye(width,at::kFloat)*.125f;}
  for(auto& [_,s]:f.initial.states)s.value=expand(s.value);
  for(auto& x:f.input)x.value=expand(x.value);
  f.graph.compile();f.initial.identity=f.graph.identity;return f;
}
GraphReference graph_reference(Fixture f,int mode,bool warm,at::ScalarType dtype) {
  return graph_reference_precision(std::move(f),mode,warm,dtype,false);
}
GraphReference graph_reference_precision(Fixture f,int mode,bool warm,at::ScalarType dtype,bool half,Options options) {
  const auto initial_cut=f.initial.cut,stop=initial_cut+11;
  const double scale=half?256.:1.;
  auto clone=[&](const Tensor& x){return x.detach().to(dtype).clone();};
  for(auto& w:f.model.nodes){w.decay=clone(w.decay);w.weight=clone(w.weight);w.bias=clone(w.bias);w.read=clone(w.read);
    for(auto& [_,x]:w.extra)x=clone(x);}
  for(auto group:{&f.model.input_scale,&f.model.agg_scale,&f.model.edge_scale,&f.model.output_scale})for(auto& x:*group)x=clone(x);
  for(auto& x:f.input)x.value=clone(x.value);for(auto& [_,s]:f.initial.states)s.value=clone(s.value);
  if(half)configure_half_reference(f);
  if(warm) {
    at::NoGradGuard guard;std::vector<External> first;for(const auto& x:f.input)if(x.time<initial_cut+2)first.push_back(x);
    Streaming cpu(f.graph,f.model,options);auto r=cpu.run(f.initial,first,initial_cut+2,initial_cut+2);
    if(half)round_half_transport(r);f.initial=std::move(r.continuation);
  }
  at::AutoGradMode grad(true);std::vector<Tensor> leaves;std::vector<std::string> names;
  auto leaf=[&](const std::string& name,Tensor& x){x=clone(x).set_requires_grad(true);leaves.push_back(x);names.push_back(name);};
  for(size_t n=0;n<f.model.nodes.size();++n) {
    auto prefix="node/"+std::to_string(n)+"/";auto& w=f.model.nodes[n];
    leaf(prefix+"decay",w.decay);leaf(prefix+"weight",w.weight);leaf(prefix+"bias",w.bias);leaf(prefix+"read",w.read);
    for(auto& [name,x]:w.extra)leaf(prefix+name,x);
  }
  const std::vector<std::string> groups{"input","aggregate","edge","output"};size_t which=0;
  for(auto group:{&f.model.input_scale,&f.model.agg_scale,&f.model.edge_scale,&f.model.output_scale}) {
    for(size_t i=0;i<group->size();++i)leaf(groups[which]+"/"+std::to_string(i),group->at(i));++which;
  }
  for(auto& [owner,s]:f.initial.states)leaf("state/"+std::to_string(owner.first)+"/"+std::to_string(owner.second),s.value);
  for(auto& a:f.initial.pending)leaf(boundary_name(a),a.value);
  std::vector<External> input;
  for(auto x:f.input)if(x.time>=f.initial.cut){leaf(boundary_name({x.batch,f.graph.inputs[x.port],x.time,0,x.port,x.position,x.value}),x.value);input.push_back(x);}
  Streaming cpu(f.graph,f.model,options);GraphReference expected;expected.result=cpu.run(f.initial,input,stop,stop);
  if(half)round_half_transport(expected.result);
  std::vector<Tensor> terms;
  if(mode==1||mode==4||mode==5)for(const auto& x:expected.result.outputs)terms.push_back(x.value.sum()*(mode==4?0.:.0625*scale));
  if(mode==2||mode==5)for(const auto& [_,s]:expected.result.continuation.states)terms.push_back(s.value.sum()*(.03125*scale));
  if(mode==3||mode==5)for(const auto& a:expected.result.continuation.pending)terms.push_back(a.value.sum()*(.015625*scale));
  std::vector<Tensor> result(leaves.size());if(!terms.empty())result=torch::autograd::grad({at::stack(terms).sum()},leaves,{},false,false,true);
  for(size_t i=0;i<names.size();++i)expected.gradients.emplace(names[i],result[i]);
  return expected;
}
void compare_graph_vjp(const ReverseTape& t,const GraphVjp& out,const GraphReference& expected) {
  compare_graph_vjp_precision(t,out,expected,false);
}
void compare_graph_vjp_precision(const ReverseTape& t,const GraphVjp& out,const GraphReference& expected,bool half) {
  const auto& g=*t.graph;
  auto check=[&](const Tensor& value,const Tensor& connected,const std::string& name){
    const auto found=expected.gradients.find(name);if(found==expected.gradients.end())throw std::runtime_error("missing CPU leaf: "+name);
    full_same_precision(value,connected,found->second,name.c_str(),half);};
  auto initial=out.initial.cpu(),ic=out.initial_connected.cpu();
  for(Index b=0;b<t.state.samples;++b)for(size_t n=0;n<g.nodes.size();++n)check(initial[b][n],ic[b][n],"state/"+std::to_string(b)+"/"+std::to_string(n));
  auto decay=out.decay.cpu(),dc=out.decay_connected.cpu(),ret=out.retention.cpu(),rc=out.retention_connected.cpu();
  auto wc=out.full_connected.cpu();auto weights=out.weights.defined()?out.weights.cpu():Tensor(),biases=out.biases.defined()?out.biases.cpu():Tensor();
  for(size_t n=0;n<g.nodes.size();++n) {
    auto prefix="node/"+std::to_string(n)+"/";check(decay[n],dc[n],prefix+"decay");
    if(weights.defined()){check(weights[n],wc[n],prefix+"weight");check(biases[n],wc[n],prefix+"bias");}
    else if(expected.gradients.at(prefix+"weight").defined()||expected.gradients.at(prefix+"bias").defined())throw std::runtime_error("identity profile lost a Full gradient");
    if(expected.gradients.count(prefix+"add_retention"))check(ret[n],rc[n],prefix+"add_retention");
    if(out.read.defined())check(out.read[n].cpu(),out.read_connected[n].cpu(),prefix+"read");
    else if(expected.gradients.at(prefix+"read").defined())throw std::runtime_error("HARD CPU Read unexpectedly connected");
  }
  auto scales=out.scales.cpu(),sc=out.scale_connected.cpu();Index offset=0;
  for(size_t i=0;i<g.inputs.size();++i,++offset)check(scales[offset],sc[offset],"input/"+std::to_string(i));
  for(size_t i=0;i<g.edges.size();++i,++offset)check(scales[offset],sc[offset],"aggregate/"+std::to_string(i));
  for(size_t slot=0;slot<g.outgoing_ports.bindings.size();++slot) {
    const auto binding=g.outgoing_ports.bindings[slot];check(scales[offset+slot],sc[offset+slot],(binding.kind?"edge/":"output/")+std::to_string(binding.id));
  }
  auto meta=out.links.messages.cpu(),valid=out.links.valid.cpu(),grads=out.messages.cpu(),on=out.message_connected.cpu();
  auto fm=t.fiber_meta.cpu(),pm=t.pending.coordinates.cpu();std::map<std::string,bool> seen;
  for(Index i=0;i<out.links.fibers+out.links.pending;++i)if(valid[i].item<bool>()&&meta[i][1].item<Index>()<0) {
    const auto row=i<out.links.fibers?fm[i]:pm[i-out.links.fibers];
    Atom a{row[0].item<Index>(),row[1].item<Index>(),row[2].item<Index>(),row[3].item<Index>(),row[4].item<Index>(),row[5].item<Index>(),{}};
    const auto name=boundary_name(a);if(seen[name])throw std::runtime_error("boundary message appeared twice");seen[name]=true;check(grads[i],on[i],name);
  }
  for(const auto& [name,gradient]:expected.gradients)if(name.rfind("boundary/",0)==0&&!seen[name])throw std::runtime_error("missing incoming boundary: "+name);
}
} // namespace tide::device_online::test
