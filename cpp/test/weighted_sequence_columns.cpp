// Independent native streaming schedule vs B/L column composition and two updates.
#include "weighted_graph_fixture.h"
#include "portable_torch/runtime.hpp"
#include "tide/weighted_sequence_columns.h"
#include "tide/stream.h"
#include "tide/parameters.h"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
namespace {
using namespace tide;
namespace col=tide::weighted::columns;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void close_value(const Tensor& a,const Tensor& b,double rtol,double atol){
  require(a.defined()==b.defined(),"column gradient connection mismatch");
  if(!a.defined())return;
  require(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type(),"column metadata mismatch");
  require(a.is_floating_point()?at::allclose(a,b,rtol,atol):at::equal(a,b),"column numerical mismatch");
}
void check(const std::string& family,bool positional=false,bool scoring=false,
           at::Device device=at::kCPU,at::ScalarType dtype=at::kDouble,const std::string& preset="native"){
  at::set_num_threads(1);
  const bool fp32=scoring||dtype==at::kFloat;
  auto close=[&](const Tensor& a,const Tensor& b){close_value(a,b,fp32?1e-5:1e-8,fp32?1e-6:1e-10);};
  auto g=weighted_check::graph("fc",family);auto m=weighted_check::model(g,family,device,dtype);
  if(positional)for(auto& node:m.nodes){auto b=weighted::node_binding(node);auto c=std::get<weighted::KVConfig>(b.memory);
    c.full_frequencies={{.17}};c.distance_slopes={.03};c.position_scale=4.;
    weighted::configure_node(node,c,b.regional,b.sources,b.aggregate_alpha);}
  ExecutionPlacement placement;placement.preset=preset;placement.scoring_dtype=scoring?"float32":"payload";
  SettleGraph spec(g,{1,2,3},"ports",4);Streaming scalar(g,place_model(g,m,placement),Options{});
  Continuation q;q.identity=g.identity;q.batch_size=2;
  std::map<Index,col::LocalState> left;
  auto owners=m.parameters().owners();std::vector<Tensor> parameters;
  for(const auto& p:owners)parameters.push_back(p.value);
  for(Index step=0;step<2;++step){
    const Index base=5*step;col::Layout layout{{0,2,4},{base,base+2,base+1,base+3}};
    auto x=at::tensor({.4,.2,-.3, .7,.3,.1, .6,-.1,.5, .9,.4,-.2},at::kDouble).to(device,dtype).reshape({4,3}).set_requires_grad(true);
    col::Message input{{0,1,2,3},x.select(1,0),x.slice(1,1)};
    std::map<Index,col::Message> ports{{0,input},{1,input}};
    std::vector<External> xs;
    for(Index b=0;b<2;++b)for(Index j=0;j<2;++j)for(Index p=0;p<2;++p)
      xs.push_back({b,p,step*2+j,layout.times[b*2+j]*4+1,x[b*2+j]});
    auto actual=col::settle(spec,m,layout,ports,left,Index{4}*1024*1024*1024,placement);
    for(const auto& [v,n]:actual.nodes){
      for(const auto& [field,t]:n.descriptors)require(t.device()==resolve_placement(placement,device).read,"column Read location mismatch");
      require(n.controls.device()==device,"column control payload owner mismatch");
    }
    auto expected=scalar.run(q,xs,(base+4)*4,(base+4)*4);
    std::vector<Tensor> aterms,eterms;
    for(const auto& out:expected.outputs){
      auto pos=(out.time-spec.rank(g.outputs[out.port]))/4;
      const auto& p=actual.outputs.at(out.port);Index row=-1;
      for(Index j=layout.offsets[out.batch];j<layout.offsets[out.batch+1];++j)if(layout.times[j]==pos)row=j;
      auto found=std::find(p.rows.begin(),p.rows.end(),row);require(found!=p.rows.end(),"missing column output");
      Index j=found-p.rows.begin();auto value=at::cat({p.strength[j].reshape({1}),p.content[j]});
      close(value,out.value);aterms.push_back(value.square().sum());eterms.push_back(out.value.square().sum());
    }
    for(const auto& [v,state]:actual.states){auto decoded=std::visit([](const auto& s){return s.states();},state);
      for(Index b=0;b<2;++b){const auto& a=decoded[b];const auto& e=expected.continuation.states.at({b,v});
        require(a.last_time==e.last_time&&a.observations==e.observations,"raw column clock mismatch");
        close(a.value,e.value);
        for(const auto& [name,t]:a.slots){close(t,e.slots.at(name));if(t.is_floating_point()){
          aterms.push_back(.13*t.square().sum());eterms.push_back(.13*e.slots.at(name).square().sum());}}
      }
    }
    auto roots=parameters;roots.push_back(x);
    auto ga=torch::autograd::grad({at::stack(aterms).sum()},roots,{},true,false,true);
    auto ge=torch::autograd::grad({at::stack(eterms).sum()},roots,{},true,false,true);
    for(size_t j=0;j<ga.size();++j)close(ga[j],ge[j]);
    // Preserve exact actual raw state while parameters advance; no replay.
    left=actual.states;for(auto& [v,s]:left)std::visit([](auto& state){state.inert=state.inert.detach();
      for(auto& [n,t]:state.fields)t=t.detach();
      if constexpr(std::is_same_v<std::decay_t<decltype(state)>,col::RecurrentState>)
        if(state.cache_values.defined())state.cache_values=state.cache_values.detach();},s);
    q=expected.continuation;for(auto& [owner,s]:q.states){s.value=s.value.detach();for(auto& [n,t]:s.slots)t=t.detach();}
    at::NoGradGuard guard;for(size_t j=0;j<parameters.size();++j)if(ga[j].defined())parameters[j].add_(ga[j],-.0001);
  }
  bool rejected=false;try{col::Layout{{0,100,1},{3}}.validate();}catch(const std::exception&){rejected=true;}
  require(rejected,"malformed layout did not reject before indexing");
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;try{
  if(argc==1){check("kv");check("kv",true);for(const auto& f:{"affine","delta","gated_delta","ssd"})check(f);
    check("kv",false,true);check("ssd",false,true);
  }else{
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    auto device=portable_torch::resolve_device(args);
    require(args.dtype==at::kFloat||args.dtype==at::kDouble,"column gate requires FP32/FP64");
    for(const auto& preset:{"mixed-a","mixed-b","mixed-c"})for(const auto& family:{"kv","affine","delta","gated_delta","ssd"}){
      check(family,false,false,device,args.dtype,device.is_cpu()?"cpu":preset);
      std::cout<<"column case passed: "<<family<<" "<<preset<<'\n';
    }
    portable_torch::synchronize(device);
  }
  runtime.close();
  std::cout<<"column standalone: independent streaming, raw state, VJP, two SGD windows passed\n";return 0;}
  catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
