// Independent named public parameter schema from architecture §2/3 and KV §2/3.
// Deliberately does not call the candidate's role-shape or packing helpers.
#include "weighted_flow_fixture.h"
#include "tide/ops.h"
namespace tide::device_online::weighted_flow_check {
std::pair<Graph,Model> configured(const weighted::RegionalConfig& r,bool query,bool sources,bool origin){
  Graph g;g.regions={{r.capacity(2),true,false,"proposal","weighted-check-"+r.selection_mode},
                   {r.capacity(2),true,false,"old","weighted-check-"+r.selection_mode}};
  g.nodes={Node{0},Node{0},Node{1},Node{1}};
  for(auto& n:g.nodes){n.memory="weighted-check";n.readout="weighted-check";n.emission="weighted-check";n.aggregation="weighted-check";}
  g.edges={{0,2,1},{0,2,1},{1,2,2},{0,3,1},{1,3,1}};g.inputs={0,1};g.outputs={2,3,2};
  if(origin)g.origins={{0,0,1}};g.compile();
  Model m;int64_t serial=0;auto parameter=[&](std::vector<int64_t> shape){int64_t n=1;for(auto x:shape)n*=x;
    return (.11*at::sin(at::arange(n,at::kFloat)*.31+.17*(++serial))).reshape(shape).set_requires_grad(true);};
  SourceClasses classes;classes.inputs={0,1};classes.edges={0,0,1,0,1};
  auto common=parameter({4,2});
  for(int64_t v=0;v<4;++v){
    NodeWeights w{at::zeros({3}),at::zeros({3,3}),at::zeros({3}),at::zeros({3})};auto& p=w.extra;
    auto put=[&](const std::string& n,std::vector<int64_t> s){p[n]=parameter(s);};
    auto head=[&](const std::string& name,int64_t in,int64_t out,bool scalar){
      put(name+(scalar?".matrix":".in_weight"),{3,in});put(name+".hidden_bias",{3});
      put(name+(scalar?".vector":".out_weight"),scalar?std::vector<int64_t>{3}:std::vector<int64_t>{out,3});
      put(name+(scalar?".bias":".out_bias"),scalar?std::vector<int64_t>{}:std::vector<int64_t>{out});};
    auto encoded=[&](const std::string& name){head(name+".encoders.cur",2,2,false);head(name+".encoders.mem",4,2,false);head(name+".encoders.stat",3,2,false);};
    weighted::KVConfig c;c.width=c.summary_width=2;c.ffn_width=3;c.adapter_rank=1;c.epsilon=.3;c.window=4;
    c.memory_read=query?"qry":"sum";if(query)c.read_frequencies={.17};c.full_frequencies={{.23}};c.distance_slopes={.09};c.position_origin=3;c.position_scale=.7;
    p["kernel.base"]=common;put("kernel.adapter_u",{4,1});put("kernel.adapter_v",{1,2});put("kernel.summary",{2,2});
    if(query)for(const auto& n:{"read_q","read_k","read_v"})put("kernel."+std::string(n),{2,2});
    for(const auto& n:{"query","out"})put("full_program.content."+std::string(n),{2,2});
    for(const auto& n:{"gate","up"})put("full_program.content."+std::string(n),{3,2});put("full_program.content.down",{2,3});
    if(r.local_score=="cat")head("read_program.score",13,1,true);
    else{const std::string name="read_program.score";encoded(name);put(name+".gate",{3,6});put(name+".up",{3,6});
      put(name+".gate_bias",{3});put(name+".up_bias",{3});put(name+".vector",{3});put(name+".bias",{});}
    if(r.region_control_mode=="context"){
      if(r.region_evidence=="cat"){put("read_program.evidence_weight",{2,13});put("read_program.evidence_bias",{2});}
      else{const std::string name="read_program.evidence";encoded(name);for(const auto& n:{"cur","mem","stat","interaction"})put(name+"."+n,{2,2});put(name+".bias",{2});}
    }
    if(r.send_gain=="context")head("read_program.gain",13,1,true);else if(r.gain_logit)put("read_program.gain_logit",{});
    if(r.region_score=="att"){
      put("read_program.current_weight",{2,2});put("read_program.current_bias",{2});
      if(r.region_key=="content")head("read_program.key",4,2,false);
      else{put("read_program.key_weight",{2,6});put("read_program.key_bias",{2});}
    }
    if(sources)for(int64_t label=0;label<2;++label){const int64_t h=3+label;const auto prefix="aggregate_program.transforms."+std::to_string(label);
      put(prefix+".a",{h,2});put(prefix+".b",{2,h});put(prefix+".embedding",{h});
      put(prefix+".a_left",{h,1});put(prefix+".a_right",{1,2});put(prefix+".b_left",{2,1});put(prefix+".b_right",{1,h});}
    weighted::configure_node(w,c,r,classes,sources?.3:0);m.nodes.push_back(std::move(w));
  }
  for(int region=0;region<2;++region){RegionWeights w;
    if(r.region_control_mode=="context")w.extra={{"head.matrix",parameter({3,4})},{"head.hidden_bias",parameter({3})},{"head.vector",parameter({3})},{"head.bias",parameter({})}};
    if(r.region_score=="att")for(const auto& n:{"in_weight","out_weight","hidden_bias","out_bias"}){
      const auto shape=std::string(n).find("weight")!=std::string::npos?std::vector<int64_t>{2,2}:std::vector<int64_t>{2};w.extra["query."+std::string(n)]=parameter(shape);}
    weighted::configure_region(w,r);m.regions.push_back(std::move(w));
  }
  for(auto collection:{&m.input_scale,&m.output_scale,&m.edge_scale,&m.agg_scale}){
    auto count=collection==&m.input_scale?g.inputs.size():collection==&m.output_scale?g.outputs.size():g.edges.size();collection->assign(count,at::ones({},at::kFloat));}
  configure_model(g,m);validate_model(g,m);return {g,m};
}
std::vector<weighted::RegionalConfig> configurations(){
  std::vector<weighted::RegionalConfig> result;int variant=0;
  for(const std::string mode:{"regional_total","regional_threshold","regional_total_zero","regional_threshold_hard",
      "constrained","hard_selection","proportional","direct_attention","unit_attention","pooled_attention"}){
    weighted::RegionalConfig c(mode);c.hidden_width=3;c.gamma_max=1.6;if(c.k)c.k=c.hard()?1:2;
    c.local_score="match";c.selection_width=2;c.alpha_match=.7;c.alpha_res=.3;c.region_score="att";
    c.region_key=(variant%2)?"content":"content_stats";c.query_weight=c.regional()?"received":"mu";c.alpha_loc=.8;c.alpha_sel=.4;
    if(c.regional())c.region_evidence="grouped";
    else if(variant%2){c.send_gain="context";c.fixed_gain.reset();}else{c.gain_logit=.3;c.fixed_gain.reset();}
    c.validate();result.push_back(c);++variant;
  }
  for(const auto& rule:{"bern","ent"}){weighted::RegionalConfig c("regional_total");c.hidden_width=3;c.k=2;c.gamma_max=1.6;
    c.region_control_mode="fixed";c.region_evidence.reset();c.total_fraction=.37;c.regularizer=rule;if(c.regularizer=="ent")c.ent_alpha=1.5;c.validate();result.push_back(c);}
  auto c=result[1];c.region_control_mode="fixed";c.region_evidence.reset();c.eta=.15;c.regularizer="ent";c.ent_alpha=1.;c.validate();result.push_back(c);
  c=result[7];c.allocation_base="received";c.query_weight="received";c.validate();result.push_back(c);
  c=result[8];c.allocation_base="mu";c.validate();result.push_back(c);return result;
}
}
