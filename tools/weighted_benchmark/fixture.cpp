#include "benchmark.h"
#include "tide/kernel.h"
#include "tide/ops.h"
namespace weighted_benchmark {
namespace {
weighted::RegionalConfig regional(const Json& x){
  if(x.at("selection_mode")!="regional_total"||x.at("local_score")!="cat"||x.at("region_score")!="ind"
      ||x.at("region_control_mode")!="context"||x.at("region_evidence")!="cat"||x.at("regularizer")!="quad")
    throw std::invalid_argument("benchmark fixture is outside the declared first-combination domain");
  weighted::RegionalConfig c;c.k=x.at("k").get<Index>();c.hidden_width=x.at("hidden_width");
  c.region_control_width=x.at("region_control_width").get<Index>();c.gamma_max=x.at("gamma_max");
  c.allocation_temperature=x.at("allocation_temperature").get<double>();c.validate();return c;
}
weighted::KVConfig kv(const Json& x){
  weighted::KVConfig c;
#define FIELD(name) c.name=x.at(#name).get<decltype(c.name)>()
  FIELD(width);FIELD(summary_width);FIELD(adapter_rank);FIELD(heads);FIELD(ffn_width);FIELD(window);FIELD(memory_read);
  FIELD(beta_short);FIELD(epsilon);FIELD(alpha_att);FIELD(alpha_ff);FIELD(position_origin);FIELD(position_scale);
  FIELD(full_frequencies);FIELD(read_frequencies);FIELD(distance_slopes);
#undef FIELD
  c.validate();return c;
}
weighted::RecurrentConfig recurrent(const Json& x){
  weighted::RecurrentConfig c;
#define FIELD(name) c.name=x.at(#name).get<decltype(c.name)>()
  FIELD(memory_family);FIELD(memory_read);FIELD(width);FIELD(summary_width);FIELD(key_width);FIELD(value_width);FIELD(ffn_width);
  FIELD(heads);FIELD(groups);FIELD(conv_size);FIELD(conv_stride);FIELD(lambda_short);FIELD(epsilon);FIELD(alpha_mem);FIELD(alpha_ff);
  FIELD(qk_epsilon);FIELD(output_epsilon);FIELD(matrix_rates);
#undef FIELD
  if(c.memory_family!="ssd")throw std::invalid_argument("benchmark recurrence must be SSD");c.validate();return c;
}
}
Fixture fixture(const Json& data){
  if(data.at("schema")!="tide-weighted-benchmark-fixture-v1")throw std::invalid_argument("unknown benchmark fixture schema");
  const auto& config=data.at("config");Fixture f;auto& g=f.graph;auto& m=f.model;
  f.span=data.at("span");f.samples=config.at("samples");f.positions=config.at("positions");
  if(f.samples<1||f.samples>16||f.positions<1||f.positions>16||f.span<1||f.span>128)
    throw std::invalid_argument("benchmark fixture exceeds bounded windows");
  const auto& graph=data.at("graph");
  for(const auto& x:graph.at("nodes")){
    Node n{x.at("region")};
#define FIELD(name) n.name=x.at(#name).get<decltype(n.name)>()
    FIELD(clear);FIELD(identity);FIELD(memory);FIELD(full);FIELD(query_heads);FIELD(kv_heads);FIELD(window);
    FIELD(emission);FIELD(emit_period);FIELD(emit_phases);FIELD(aggregation);FIELD(readout);FIELD(next_state);
#undef FIELD
    const auto& clock=x.at("state_clock");n.state_clock={clock.at("period"),clock.at("first"),clock.at("count")};g.nodes.push_back(n);
  }
  for(const auto& x:graph.at("edges"))g.edges.push_back({x.at("source"),x.at("target"),x.at("delay")});
  for(const auto& x:graph.at("regions"))g.regions.push_back({x.at("budget"),x.at("observe_all"),x.at("count_priority"),x.at("read_mode"),x.at("selector")});
  g.inputs=graph.at("inputs").get<std::vector<Index>>();g.outputs=graph.at("outputs").get<std::vector<Index>>();
  const auto& l=graph.at("layout");g.layout=PortLayout{l.at("edge_source"),l.at("edge_target"),l.at("input"),l.at("output")};
  const auto& d=graph.at("domain");g.source_domain=SourceDomain{d.at("edge_target"),d.at("input")};g.compile();
  if(g.identity!=graph.at("native_identity"))throw std::invalid_argument("decoded fixture graph identity differs");
  std::vector<Tensor> values;
  for(const auto& x:data.at("tensors")){
    if(x.at("dtype")!="float32")throw std::invalid_argument("finite benchmark requires explicit FP32");
    values.push_back(at::tensor(x.at("values").get<std::vector<float>>(),at::kFloat)
      .reshape(x.at("shape").get<std::vector<Index>>()).set_requires_grad(x.at("trainable").get<bool>()));
  }
  auto tensor=[&](const Json& i){return values.at(i.get<size_t>());};
  const auto& classes=data.at("classes");SourceClasses sources{classes.at("edges"),classes.at("inputs")};
  for(size_t v=0;v<g.nodes.size();++v){const auto& x=data.at("nodes").at(v);
    NodeWeights n{tensor(x.at("decay")),tensor(x.at("weight")),tensor(x.at("bias")),tensor(x.at("read"))};
    for(const auto& [name,i]:x.at("extra").items())n.extra[name]=tensor(i);
    const auto& memory=data.at("memories").at(v);const auto region=regional(data.at("regions").at(g.nodes[v].region));
    if(config.at("family")=="kv")weighted::configure_node(n,kv(memory),region,sources,data.at("alpha_agg"));
    else if(config.at("family")=="ssd")weighted::configure_node(n,recurrent(memory),region,sources,data.at("alpha_agg"));
    else throw std::invalid_argument("unknown finite benchmark memory");m.nodes.push_back(std::move(n));
  }
  for(size_t r=0;r<g.regions.size();++r){RegionWeights w;
    for(const auto& [name,i]:data.at("region_weights").at(r).items())w.extra[name]=tensor(i);
    weighted::configure_region(w,regional(data.at("regions").at(r)));m.regions.push_back(std::move(w));}
  const auto& scales=data.at("scales");
  for(const auto& i:scales.at("input_scale"))m.input_scale.push_back(tensor(i));
  for(const auto& i:scales.at("output_scale"))m.output_scale.push_back(tensor(i));
  for(const auto& i:scales.at("edge_scale"))m.edge_scale.push_back(tensor(i));
  for(const auto& i:scales.at("agg_scale"))m.agg_scale.push_back(tensor(i));
  configure_model(g,m);validate_model(g,m);
  for(const auto& x:data.at("inputs"))f.inputs.push_back({x.at("batch"),x.at("port"),x.at("position"),x.at("time"),tensor(x.at("value"))});
  return f;
}
std::vector<External> inputs(const Fixture& f,Index step){
  std::vector<External> out;std::map<const void*,Tensor> copies;
  for(auto x:f.inputs){const auto id=x.value.unsafeGetTensorImpl();
    if(!copies.count(id))copies[id]=x.value.detach().clone().set_requires_grad(true);
    x.value=copies.at(id);x.time+=step*f.span;x.position+=step*f.positions;out.push_back(x);}
  return out;
}
}
