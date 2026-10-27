// Pinned FC §1 / HB §1 depth=1, width=2; feedback/nonaligned counterexamples.
#include "weighted_graph_fixture.h"
#include "tide/kernel.h"
#include <set>
namespace weighted_check {
using namespace tide;using namespace tide::weighted;
Graph graph(const std::string& topology,const std::string& family){
  Graph g;Index regions=topology=="hb"?4:topology=="feedback"?2:3;
  for(Index r=0;r<regions;++r){g.regions.push_back({2,true,false,"proposal","weighted-core5-total-context-cat-quad"});
    for(Index j=0;j<2;++j){Node n{r};n.memory="weighted-core5-check-"+family;n.readout="weighted-core5-check-cat";
      n.emission="weighted-core5-check-full";n.aggregation="weighted-core5-check-source";g.nodes.push_back(n);}}
  auto connect=[&](Index a,Index b,bool diagonal){for(Index s=0;s<2;++s)for(Index t=0;t<2;++t)
    if(!diagonal||s==t)g.edges.push_back({2*a+s,2*b+t,1});};
  if(topology=="hb"){connect(0,1,true);connect(0,2,true);connect(1,3,true);connect(2,3,true);}
  else for(Index r=0;r<regions-1;++r)connect(r,r+1,false);
  if(topology=="nonaligned")for(size_t i=0;i<g.edges.size();++i)g.edges[i].delay=1+(i%3);
  if(topology=="feedback"){g.edges.push_back({2,0,2});g.edges.push_back({3,1,3});g.edges.push_back({0,2,1});}
  g.inputs={0,1};g.outputs={2*(regions-1),2*(regions-1)+1};g.compile();return g;
}
Model model(const Graph& g,const std::string& family,at::Device device,at::ScalarType dtype){
  Model m;const auto o=at::TensorOptions().dtype(dtype);Index serial=0;
  auto parameter=[&](std::vector<Index> shape){Index n=1;for(auto d:shape)n*=d;
    return (.12*at::sin(at::arange(n,o)*.7+(++serial)*.3)).reshape(shape).to(device).set_requires_grad(true);};
  RegionalConfig r;r.k=2;r.hidden_width=3;r.region_control_width=2;r.gamma_max=2;
  SourceClasses classes;for(const auto& e:g.edges)classes.edges.push_back(e.source+1);classes.inputs=std::vector<Index>(g.inputs.size(),0);
  std::map<Index,Tensor> bases,as,bs;std::map<std::pair<Index,Index>,Tensor> embeddings;
  for(Index region=0;region<static_cast<Index>(g.regions.size());++region){bases[region]=parameter({4,2});as[region]=parameter({3,2});bs[region]=parameter({2,3});}
  const std::vector<std::string> families{"kv","affine","delta","gated_delta","ssd"};
  for(size_t i=0;i<g.nodes.size();++i){
    auto region=g.nodes[i].region;auto kind=family=="heterogeneous"?families[i%families.size()]:family;
    NodeWeights w{at::zeros({3},o).to(device),at::zeros({3,3},o).to(device),at::zeros({3},o).to(device),at::zeros({3},o).to(device)};
    auto& p=w.extra;auto add=[&](const std::string& name,std::vector<Index> shape){p[name]=parameter(shape);};
    add("kernel.summary",{2,2});
    for(const std::string name:{"gate","up"})add("full_program.content."+name,{3,2});
    add("full_program.content.down",{2,3});
    if(kind=="kv"){
      KVConfig c;c.width=c.summary_width=2;c.adapter_rank=1;c.ffn_width=3;c.window=7;c.epsilon=.3;
      p["kernel.base"]=bases.at(region);add("kernel.adapter_u",{4,1});add("kernel.adapter_v",{1,2});
      add("full_program.content.query",{2,2});add("full_program.content.out",{2,2});
      configure_node(w,c,r,classes,.3);
    }else{
      RecurrentConfig c;c.memory_family=kind;c.width=c.summary_width=c.key_width=c.value_width=2;c.ffn_width=3;
      c.epsilon=.3;c.memory_read="state";
      if(c.basic()){
        add("kernel.key",{2,2});add("kernel.value",{2,2});add("kernel.retain",{2});add("kernel.retain_bias",{});
        if(kind=="delta"){add("kernel.write",{2});add("kernel.write_bias",{});}
        add("kernel.read_query",{2,2});add("kernel.read_out",{2,2,2});
        add("full_program.content.query",{2,2});add("full_program.content.out",{2,2,2});
      }else{
        c.heads=2;c.groups=1;c.conv_size=3;c.conv_stride=2;c.matrix_rates={.03,.0};
        add("kernel.conv.weight",{3,c.conv_width()});add("kernel.conv.bias",{c.conv_width()});
        add("kernel.read_out",{2,4});add("kernel.log_rate",{2});add("kernel.delta_bias",{2});
        add("full_program.content.out",{2,4});
        if(kind=="gated_delta"){
          add("kernel.projection",{c.conv_width(),2});add("kernel.delta",{2,2});add("kernel.write",{2,2});add("kernel.write_bias",{2});
          add("full_program.content.rho",{2,2});add("full_program.content.output_gate",{4,2});
        }else{
          add("kernel.projection",{4+c.conv_width()+2,2});add("full_program.content.direct",{2});add("full_program.content.rho",{4});
        }
      }
      configure_node(w,c,r,classes,.3);
    }
    add("read_program.score.matrix",{3,13});add("read_program.score.hidden_bias",{3});
    add("read_program.score.vector",{3});add("read_program.score.bias",{});
    add("read_program.evidence_weight",{2,13});add("read_program.evidence_bias",{2});
    std::set<Index> labels;for(size_t j=0;j<g.inputs.size();++j)if(g.inputs[j]==static_cast<Index>(i))labels.insert(0);
    for(size_t e=0;e<g.edges.size();++e)if(g.edges[e].target==static_cast<Index>(i))labels.insert(classes.edges[e]);
    for(auto label:labels){auto name="aggregate_program.transforms."+std::to_string(label)+".";
      auto id=std::make_pair(region,label);if(!embeddings.count(id))embeddings[id]=parameter({3});
      p[name+"a"]=as.at(region);p[name+"b"]=bs.at(region);p[name+"embedding"]=embeddings.at(id);}
    m.nodes.push_back(w);
  }
  for(size_t i=0;i<g.regions.size();++i){RegionWeights w;w.extra={{"head.matrix",parameter({3,4})},{"head.hidden_bias",parameter({3})},
      {"head.vector",parameter({3})},{"head.bias",parameter({})}};configure_region(w,r);m.regions.push_back(w);}
  for(size_t i=0;i<g.inputs.size();++i)m.input_scale.push_back(at::ones({},o).to(device));
  for(size_t i=0;i<g.outputs.size();++i)m.output_scale.push_back(at::ones({},o).to(device));
  for(size_t i=0;i<g.edges.size();++i){m.edge_scale.push_back(at::ones({},o).to(device));m.agg_scale.push_back(at::ones({},o).to(device));}
  configure_model(g,m);return m;
}
std::vector<External> inputs(const Graph& g,at::Device device,at::ScalarType dtype,Index start){
  std::vector<External> xs;
  for(Index b=0;b<2;++b)for(Index pos=0;pos<2;++pos){
    auto h=at::tensor({1.,.2+.1*b,-.3+.2*pos},at::TensorOptions().dtype(dtype)).to(device).set_requires_grad(true);
    for(size_t port=0;port<g.inputs.size();++port)xs.push_back({b,static_cast<Index>(port),pos,start+1+5*pos,h});}
  return xs;
}
}
