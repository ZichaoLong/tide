// Integration oracle uses installed scalar functions and an independent CPU
// schedule. Local mathematical anchors remain the separately qualified formulas.
#include "weighted_stage_check.h"
#include "weighted_flow_fixture.h"
#include "../../cpp/test/weighted_graph_fixture.h"
#include "tide/kernel.h"
#include <ATen/core/grad_mode.h>
namespace tide::device_online::stage_check {
Fixture fixture(const std::string& family,const std::string& read){
  Fixture f;auto& g=f.graph;g.regions={{2,true,false,read,"weighted-core5-total-context-cat-quad"}};
  for(int j=0;j<2;++j){Node n{0};n.memory="weighted-core5-check-"+family;n.readout="weighted-core5-check-cat";
    n.emission="weighted-core5-check-full";n.aggregation="weighted-core5-check-source";g.nodes.push_back(n);}
  g.inputs={0,0,1};g.outputs={0,1,0};
  if(family=="kv_query"||family=="kv_node_gain"||family=="kv_attention_gain"){
    const auto configs=weighted_flow_check::configurations();const auto r=configs[family=="kv_query"?0:family=="kv_node_gain"?6:7];
    auto source=weighted_flow_check::configured(r,true,true,false);g.regions[0].budget=r.capacity(2);g.regions[0].selector="weighted-stage-"+r.selection_mode;
    f.model.nodes={source.second.nodes[0],source.second.nodes[1]};f.model.regions={source.second.regions[0]};
    SourceClasses classes;classes.inputs={0,1,0};
    for(auto& w:f.model.nodes){const auto binding=weighted::node_binding(w);weighted::configure_node(w,std::get<weighted::KVConfig>(binding.memory),r,classes,.3);}
    f.model.input_scale.assign(3,at::ones({},at::kFloat));f.model.output_scale.assign(3,at::ones({},at::kFloat));
    g.compile();configure_model(g,f.model);
  }else{g.compile();f.model=weighted_check::model(g,family,at::kCPU,at::kFloat);}
  f.initial.identity=g.identity;f.initial.batch_size=rows;f.initial.cut=3;
  at::NoGradGuard no_grad;
  for(int64_t b=0;b<rows;++b)for(int64_t v=0;v<2;++v){auto s=f.model.nodes[v].kernel->initial(f.model.nodes[v]);
    s.last_time=2;s.observations=2;
    if(family.rfind("kv",0)==0){
      s.slots["times"]=at::tensor(std::vector<int64_t>{0,2},at::kLong);
      s.slots["weights"]=at::tensor({.3f,.7f},at::kFloat);
      s.slots["keys"]=at::tensor({.1f,-.2f,.3f,.2f},at::kFloat).reshape({2,2});
      s.slots["values"]=at::tensor({.3f,.1f,-.2f,.4f},at::kFloat).reshape({2,2});
      s.slots["summaries"]=at::tensor({.4f,.1f,.2f,-.3f},at::kFloat).reshape({2,2});
    }else{for(auto& [name,x]:s.slots)if(x.scalar_type()==at::kFloat)x.fill_(name=="W"?.7:.09*(1+b+v));
      s.slots["time"]=at::scalar_tensor(2,at::kLong);
      if(family=="gated_delta"||family=="ssd"){
        const auto rc=std::get<weighted::RecurrentConfig>(weighted::node_binding(f.model.nodes[v]).memory);
        s.slots["conv_times"]=at::tensor(std::vector<int64_t>{0,2},at::kLong);
        s.slots["conv_values"]=at::full({2,rc.conv_width()},.12+.03*v,at::kFloat);
      }
    }
    s.value=s.value.detach().clone().set_requires_grad(true);
    for(auto& [name,x]:s.slots)if(x.scalar_type()==at::kFloat)x=x.detach().clone().set_requires_grad(true);
    f.initial.states[{b,v}]=s;
  }
  for(int64_t b=0;b<3;++b)for(int64_t port=0;port<3;++port){if(b==1&&port==2)continue;
    auto x=at::tensor({b==1?0.f:.2f+.13f*port,.15f+.07f*b,-.23f+.11f*port},at::kFloat).set_requires_grad(true);
    f.inputs.push_back({b,port,0,b==2?11:5,x});}
  return f;
}
AtomBatch atoms(const Fixture& f,at::Device d){
  auto meta=at::zeros({capacity,6},at::kLong),values=at::zeros({capacity,3},at::kFloat),valid=at::zeros({capacity},at::kBool);
  for(size_t j=0;j<f.inputs.size();++j){const auto& x=f.inputs[j];meta[j]=at::tensor(std::vector<int64_t>{x.batch,f.graph.inputs[x.port],x.time,0,x.port,x.position},at::kLong);
    values[j].copy_(x.value);valid[j]=true;}
  return {meta.to(d),values.to(d),valid.to(d)};
}
WeightedNodeRoots roots(const WeightedFlowNode& n){
  auto row=[](const at::Tensor& x){return WeightedRowAdjoint{at::zeros_like(x),at::zeros({rows},x.options().dtype(at::kBool))};};
  WeightedNodeRoots out;out.content=row(n.aggregate->output);out.full=row(n.full);
  out.proposal={at::zeros_like(n.proposal.floating),at::zeros({rows,int64_t(n.layout.fields().size())},n.input.present.options())};
  for(const auto& [name,x]:n.describe->fields)out.describe[name]=row(x);return out;
}
namespace {
at::Tensor values(at::IntArrayRef shape,int64_t b){int64_t size=1;for(auto n:shape)size*=n;
  return b==2?at::zeros(shape,at::kFloat):(.3*at::sin(at::arange(size,at::kFloat)*.17+.3)).reshape(shape);}
void set(const WeightedRowAdjoint& root,const at::Tensor& present,bool selected){
  auto flag=present.cpu();root.value.fill_(NAN);root.connected.zero_();if(!selected)return;
  for(int64_t b=0;b<rows;++b)if(flag[b].item<bool>()){root.value[b].copy_(values(root.value[b].sizes(),b));root.connected[b]=true;}
}
}
void set_roots(const WeightedFlowNodes& nodes,const std::vector<WeightedNodeRoots>& roots,
    const std::map<int64_t,WeightedRowAdjoint>& controls,const std::string& kind){
  for(size_t v=0;v<nodes.size();++v){const auto& n=*nodes[v];const auto& root=roots[v];
    set(root.content,n.input.present,kind=="content"||kind=="all");set(root.full,n.active,kind=="full"||kind=="all");
    set(controls.at(v),kind=="full"?n.active:n.input.present,kind=="control"||kind=="full"||kind=="all");
    for(const auto& [name,g]:root.describe)set(g,n.input.present,kind=="describe."+name||kind=="all");
    root.proposal.value.fill_(NAN);root.proposal.connected.zero_();
    for(size_t j=0;j<n.layout.fields().size();++j){const auto& field=n.layout.fields()[j];if(kind!="raw."+field.name&&kind!="all")continue;
      const auto width=field.width*(field.ragged?n.layout.capacity():1);
      for(int64_t b=0;b<rows;++b)root.proposal.value[b].narrow(0,field.offset,width).copy_(values({width},b));root.proposal.connected.select(1,j).fill_(true);
    }
  }
}
}
