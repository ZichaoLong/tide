#include "consumer.h"
#include "source_values.h"
#include <tide/ops.h>
#include <tide/kernel.h>
#include <ATen/core/grad_mode.h>
#include <ATen/Parallel.h>
#include <stdexcept>

namespace tide_flow {
Tensor source_values(const std::string& name,const std::vector<Index>& shape,Index seed) {
  const NamedSource source(name,seed);
  auto result=at::empty(shape,at::TensorOptions().device(at::kCPU).dtype(at::kFloat));
  auto* values=result.data_ptr<float>();
  at::parallel_for(0,result.numel(),32768,[&](Index begin,Index end) {
    for(Index i=begin;i<end;++i)values[i]=source(i);
  });
  return result;
}
Fixture fixture(const Packet& p,const Config& c,at::Device device) {
  at::NoGradGuard no_grad;Fixture f;auto g=p.body;auto& m=f.model;
  const auto d=p.width;const auto opts=at::TensorOptions().device(device).dtype(c.runtime.dtype);
  auto zero=at::zeros({d},opts),matrix=at::zeros({d,d},opts),identity=at::eye(d,opts),one=at::ones({},opts);
  auto parameter=[&](std::string name,const std::vector<Index>& shape,bool ones=false) {
    auto value=ones?at::ones(shape,at::kFloat):source_values(name,shape,p.seed);
    return value.to(opts).set_requires_grad(true);
  };
  for(Index v=0;v<Index(g.nodes.size());++v) {
    tide::NodeWeights w{zero,matrix,zero,zero};const auto prefix="node/"+std::to_string(v)+"/";
    w.extra["lh_norm_weight"]=parameter(prefix+"norm",{d},true);
    if(p.memory=="add") {
      w.extra["add_retention"]=at::scalar_tensor(.99,at::kFloat).to(opts);
      for(Index s=0;s<g.source_counts[v];++s)w.extra["agg_logit_"+std::to_string(s)]=parameter(prefix+"agg_logit_"+std::to_string(s),{},true);
    } else {
      w.extra["fiber_qkv"]=parameter(prefix+"fiber_qkv",{d,3*d});
      w.extra["fiber_out"]=parameter(prefix+"fiber_out",{d,d});
      w.extra["fiber_qkv_bias"]=at::zeros({3*d},opts);w.extra["fiber_out_bias"]=zero;
      w.extra["fiber_decay"]=at::scalar_tensor(.01,at::kFloat).to(opts);
      w.extra["fiber_pool"]=parameter(prefix+"fiber_pool",{g.source_counts[v]},true);
    }
    const auto begin=g.outgoing_ports.offsets[v],end=g.outgoing_ports.offsets[v+1];
    for(Index j=begin;j<end;++j) {
      const auto binding=g.outgoing_ports.bindings[j];const auto slot=std::to_string(j-begin);
      w.extra["emit_w_"+slot]=binding.kind?parameter("edge/"+std::to_string(binding.id)+"/projection",{d,d}):identity;
      w.extra["emit_b_"+slot]=zero;
    }
    m.nodes.push_back(std::move(w));
  }
  m.regions.resize(g.regions.size());m.input_scale.assign(g.inputs.size(),one);m.output_scale.assign(g.outputs.size(),one);
  m.agg_scale.assign(g.edges.size(),one);m.edge_scale.assign(g.edges.size(),one);
  tide::configure_model(g,m);tide::validate_model(g,m);
  if(p.topology=="ranked-local") {
    f.settle=std::make_unique<tide::SettleGraph>(g,p.ranks);
    f.graph=f.settle->encoded_graph();f.model=f.settle->embed_model(m);
  } else {
    if(c.family=="settle")throw std::invalid_argument("delayed graph has no Settle equivalence");
    f.graph=g;auto& encoded=f.graph;const Index n=g.nodes.size(),r=g.regions.size(),e=g.edges.size();
    encoded.nodes.push_back({r,false,true});encoded.nodes.push_back({r+1,false,true});
    encoded.regions.push_back({1});encoded.regions.push_back({1});
    for(Index i=0;i<Index(g.inputs.size());++i) {
      encoded.edges.push_back({n,g.inputs[i],1});encoded.layout->edge_source.push_back(i);
      encoded.layout->edge_target.push_back(g.layout->input[i]);encoded.source_domain->edge_target.push_back(g.source_domain->input[i]);
      encoded.origins.push_back({e+i,i,p.stride});
    }
    for(Index i=0;i<Index(g.outputs.size());++i) {
      encoded.edges.push_back({g.outputs[i],n+1,1});encoded.layout->edge_source.push_back(g.layout->output[i]);
      encoded.layout->edge_target.push_back(i);encoded.source_domain->edge_target.push_back(i);
    }
    encoded.inputs={n};encoded.outputs={n+1};encoded.layout->input={0};encoded.layout->output={0};encoded.source_domain->input={0};
    encoded.compile();
    m.nodes.push_back({zero,matrix,zero,zero});m.nodes.push_back({zero,matrix,zero,zero});m.regions.resize(r+2);
    m.agg_scale.insert(m.agg_scale.end(),m.input_scale.begin(),m.input_scale.end());
    m.agg_scale.insert(m.agg_scale.end(),m.output_scale.begin(),m.output_scale.end());
    m.edge_scale.resize(encoded.edges.size(),one);m.input_scale={one};m.output_scale={one};
    tide::configure_model(encoded,m);tide::validate_model(encoded,m);
  }
  f.embedding=parameter("embedding",{p.vocab,d});f.head=parameter("head",{p.vocab,d});
  f.parameters.add_model(f.model);f.parameters.add("embedding",f.embedding);f.parameters.add("head",f.head);
  Index count=0;for(const auto& owner:f.parameters.owners())count+=owner.value.numel();
  if(count!=p.parameters())throw std::logic_error("materialized model parameter count disagrees with packet");
  return f;
}
} // namespace tide_flow
