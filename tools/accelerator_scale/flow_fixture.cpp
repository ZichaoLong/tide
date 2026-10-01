#include "flow_fixture.h"
#include <tide/kernel.h>
#include <tide/lazy_add.h>
#include <tide/fiber_attention.h>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace accelerator_scale::flows {
Topology read_topology(const std::string& path) {
  std::ifstream in(path);std::string magic;Index n=0,r=0,e=0,ni=0,no=0,budget=0,clear=0;
  Topology t;in>>magic>>n>>r>>e>>ni>>no>>t.period>>budget>>clear;
  if(!in || magic!="TIDE_COMPLETE_FLOW_1" || n<2 || n>200000 || r<2 || r>n
      || e<1 || e>4000000 || ni<1 || ni>n || no<1 || no>n || t.period<3
      || t.period>10000000 || budget<1 || budget>n || (clear!=0 && clear!=1))
    throw std::invalid_argument("invalid complete-flow topology header");
  auto& g=t.body;
  for(Index i=0;i<n;++i){Index region=-1;in>>region;
    if(!in || region<0 || region>=r)throw std::invalid_argument("invalid body region");
    g.nodes.push_back(tide::Node{region,bool(clear)});}
  std::set<Index> ranks;
  for(Index i=0;i<r;++i){Index rank=0;in>>rank;
    if(!in || rank<1 || rank>=t.period-1 || !ranks.insert(rank).second)
      throw std::invalid_argument("invalid body region rank");
    t.ranks.push_back(rank);g.regions.push_back({budget});}
  for(auto [count,ports]:std::vector<std::pair<Index,std::vector<Index>*>>{{ni,&g.inputs},{no,&g.outputs}})
    for(Index i=0;i<count;++i){Index node=-1;in>>node;
      if(!in || node<0 || node>=n)throw std::invalid_argument("invalid body boundary");
      ports->push_back(node);}
  t.rank_aligned=true;
  for(Index i=0;i<e;++i){tide::Edge edge;in>>edge.source>>edge.target>>edge.delay;
    if(!in || edge.source<0 || edge.target<0 || edge.source>=n || edge.target>=n || edge.delay<1 || edge.delay>=t.period)
      throw std::invalid_argument("invalid body physical edge");
    t.rank_aligned &= edge.delay==t.ranks[g.nodes[edge.target].region]-t.ranks[g.nodes[edge.source].region];
    g.edges.push_back(edge);}
  if(in>>magic)throw std::invalid_argument("trailing complete-flow topology data");
  g.compile();const auto order=g.topological_order();
  std::vector<bool> reached(n),useful(n);for(auto i:g.inputs)reached[i]=true;for(auto i:g.outputs)useful[i]=true;
  for(auto i:order)for(Index j=g.csr.offsets[i];j<g.csr.offsets[i+1];++j)
    reached[g.edges[g.csr.edges[j]].target]=reached[g.edges[g.csr.edges[j]].target] || reached[i];
  for(auto it=order.rbegin();it!=order.rend();++it)for(Index j=g.csr.offsets[*it];j<g.csr.offsets[*it+1];++j)
    useful[*it]=useful[*it] || useful[g.edges[g.csr.edges[j]].target];
  for(Index i=0;i<n;++i)if(!reached[i] || !useful[i])throw std::invalid_argument("inactive-scale body node");
  if(t.rank_aligned && t.period!=*ranks.rbegin()+2)throw std::invalid_argument("Settle stride mismatch");
  return t;
}

Fixture fixture(const pdg_scale::Config& c,const Topology& t,bool row_emission) {
  at::NoGradGuard guard;Fixture out;out.period=t.period;out.body=t.body;
  auto& g=out.body;auto& m=out.body_model;auto& f=out.values;
  const auto width=c.width,n=Index(g.nodes.size());
  auto opts=at::TensorOptions().dtype(c.runtime.dtype).device(at::kCPU);
  auto zero=at::zeros({width},opts),dummy=at::zeros({width,width},opts),one=at::ones({},opts),identity=at::eye(width,opts);
  auto scalar=[&](double x){auto v=at::scalar_tensor(x,opts);
    return c.quantized_fp16_reference?v.to(at::kHalf).to(c.runtime.dtype):v;};
  auto parameter=[&](std::vector<Index> shape,bool random=true) {
    auto value=at::empty(shape,opts.dtype(c.runtime.dtype==at::kHalf?at::kFloat:c.runtime.dtype));
    if(random)value.normal_(0,.02);else value.fill_(1);
    if(c.runtime.dtype==at::kHalf || c.quantized_fp16_reference)value=value.to(at::kHalf).to(c.runtime.dtype);
    value.set_requires_grad(true);f.owners.push_back(value);return value;
  };
  std::vector<Index> outgoing(n),edge_rows;
  for(const auto& edge:g.edges)edge_rows.push_back(outgoing[edge.source]++);
  for(Index i=0;i<n;++i) {
    auto& node=g.nodes[i];
    node.memory=c.memory=="add"?"lh-add-repeat-v1":"lh-fiber-attention-all-softmax-repeat-v1";
    node.aggregation=c.memory=="add"?"all_softmax":"sum";
    node.query_heads=node.kv_heads=4;node.full="lh-silu-rms-v1";node.readout="norm-fp64-v1";
    node.emission=row_emission?"broadcast":"slot_affine";
  }
  g.compile();
  for(Index i=0;i<n;++i) {
    tide::NodeWeights w{zero,dummy,zero,zero};const auto& node=g.nodes[i];
    if(c.memory=="add") {
      w.kernel=tide::make_add_repeat_kernel();w.extra["add_retention"]=scalar(.99);
      auto pool=parameter({g.source_counts[i]},false);at::AutoGradMode track(true);
      for(Index slot=0;slot<g.source_counts[i];++slot)w.extra["agg_logit_"+std::to_string(slot)]=pool[slot];
    } else {
      w.kernel=tide::make_fiber_attention_kernel(node,g.source_counts[i],c.attention_packing,c.fiber_pooling,c.fiber_cache,c.attention_layout);
      w.extra["fiber_qkv"]=parameter({width,3*width});w.extra["fiber_out"]=parameter({width,width});
      w.extra["fiber_qkv_bias"]=at::zeros({3*width},opts);w.extra["fiber_out_bias"]=zero;
      w.extra["fiber_decay"]=scalar(.01);w.extra["fiber_pool"]=parameter({g.source_counts[i]},false);
    }
    w.extra["lh_norm_weight"]=parameter({width},false);
    auto weight=outgoing[i]?parameter({outgoing[i]*width,width}):Tensor();
    std::vector<Index> rows;
    for(Index j=g.outgoing_ports.offsets[i];j<g.outgoing_ports.offsets[i+1];++j) {
      const auto binding=g.outgoing_ports.bindings[j];rows.push_back(binding.kind?edge_rows[binding.id]:-1);
    }
    if(row_emission) {
      if(weight.defined())w.extra["row_emit_weight"]=weight;
      w.full_kernel=tide::make_row_emit(rows,std::vector<Index>(rows.size(),0),1,outgoing[i]);
    } else {
      at::AutoGradMode track(true);
      for(size_t slot=0;slot<rows.size();++slot) {
        const auto row=rows[slot];
        w.extra["emit_w_"+std::to_string(slot)]=row<0?identity:weight.slice(0,row*width,(row+1)*width).t();
        w.extra["emit_b_"+std::to_string(slot)]=zero;
      }
    }
    m.nodes.push_back(std::move(w));
  }
  m.input_scale.assign(g.inputs.size(),one);m.output_scale.assign(g.outputs.size(),one);
  m.agg_scale.assign(g.edges.size(),one);m.edge_scale=m.agg_scale;
  if(t.rank_aligned) {
    out.settle=std::make_shared<tide::SettleGraph>(g,t.ranks);
    f.graph=out.settle->encoded_graph();f.model=out.settle->embed_model(m);
  } else {
    // The same input broadcast and output-sum boundaries, with nonuniform body
    // arrivals. This explicit DAG encoding has no Settle rank-alignment claim.
    f.graph=g;f.model=m;const auto r=Index(g.regions.size());
    auto& enc=f.graph;enc.nodes.push_back(tide::Node{r,false,true});enc.nodes.push_back(tide::Node{r+1,false,true});
    enc.regions.push_back({1});enc.regions.push_back({1});
    auto& layout=*enc.layout;auto& domain=*enc.source_domain;
    for(size_t p=0;p<g.inputs.size();++p) {
      enc.origins.push_back({Index(enc.edges.size()),Index(p),out.period});
      enc.edges.push_back({n,g.inputs[p],1});layout.edge_source.push_back(p);layout.edge_target.push_back(g.layout->input[p]);domain.edge_target.push_back(g.source_domain->input[p]);
    }
    for(size_t p=0;p<g.outputs.size();++p){enc.edges.push_back({g.outputs[p],n+1,1});
      layout.edge_source.push_back(g.layout->output[p]);layout.edge_target.push_back(p);domain.edge_target.push_back(p);}
    enc.inputs={n};enc.outputs={n+1};layout.input={0};layout.output={0};domain.input={0};enc.compile();
    tide::configure_model(g,f.model);
    for(int i=0;i<2;++i){f.model.nodes.push_back({zero,dummy,zero,zero});f.model.regions.push_back({});}
    f.model.input_scale=f.model.output_scale={one};
    f.model.agg_scale.insert(f.model.agg_scale.end(),m.input_scale.begin(),m.input_scale.end());
    f.model.agg_scale.insert(f.model.agg_scale.end(),m.output_scale.begin(),m.output_scale.end());
    f.model.edge_scale.resize(enc.edges.size(),one);
    tide::configure_model(enc,f.model);
  }
  out.edge_rows=edge_rows;out.edge_rows.resize(f.graph.edges.size(),-1);
  f.embedding=parameter({c.vocab,width});f.head=parameter({c.vocab,width});
  Index count=0;for(const auto& p:f.owners)count+=p.numel();
  const auto expected=Index(g.edges.size())*width*width+n*width+Index(g.edges.size()+g.inputs.size())+2*c.vocab*width
    +(c.memory=="attention"?4*n*width*width:0);
  if(count!=expected)throw std::logic_error("active-scale parameter owner accounting mismatch");
  f.inventory={{"parameters",double(count)},{"body_nodes",double(n)},{"reachable_nodes",double(n)},
    {"body_edges",double(g.edges.size())},{"physical_edges",double(f.graph.edges.size())},
    {"pdg_nodes",double(f.graph.nodes.size())},{"parameter_owners",double(f.owners.size())}};
  return out;
}

bounded::Schedule schedule(const Fixture& f,const bounded::Limits& limits,bool training) {
  const Index n=f.values.graph.nodes.size(),e=f.values.graph.edges.size(),d=f.values.model.width();
  const long double b=limits.batch,t=limits.tokens,s=c10::elementSize(f.values.embedding.scalar_type());
  const long double deps=limits.connectivity?f.values.owners.size()+limits.tokens:1;
  long double bound=16*b*(n+e)*(d*s+deps)*(limits.trace || training?t:2);
  if(f.body.nodes[0].memory!="lh-add-repeat-v1")bound+=8*b*e*t*d*s;
  if(bound>std::numeric_limits<int64_t>::max())throw std::overflow_error("flow workspace bound overflow");
  return {f.period,f.edge_rows,int64_t(bound)};
}
}  // namespace accelerator_scale::flows
