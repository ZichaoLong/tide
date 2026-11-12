// Independent native streaming schedule vs B/L column composition and two updates.
#include "weighted_graph_fixture.h"
#include "portable_torch/runtime.hpp"
#include "tide/weighted_sequence_columns.h"
#include "tide/stream.h"
#include "tide/parameters.h"
#include "tide/ownership.h"
#include "tide/kernel.h"
#include "../src/weighted_column_sources.h"
#include <cstdlib>
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
void check_source_groups(at::Device device=at::kCPU,at::ScalarType dtype=at::kDouble){
  auto leaf=[&](std::vector<int64_t> shape,double offset){Index n=1;for(auto d:shape)n*=d;
    return (at::arange(n,at::kDouble)*.03+offset).reshape(shape).to(device,dtype).set_requires_grad(true);};
  std::vector<col::SourceRequest> requests;
  for(Index i=0;i<3;++i)requests.push_back({leaf({3,2},.1+i*.04),
    {leaf({2,2},.2),leaf({2,2},-.1),leaf({2},.1),leaf({2,1},.12),leaf({1,2},.13),leaf({2,1},-.07),leaf({1,2},.09)},.4,.1});
  requests[1].parameters[0]=requests[0].parameters[0];requests[1].parameters[3]=requests[0].parameters[3];
  std::vector<Tensor> roots;
  for(const auto& r:requests){roots.push_back(r.content);for(const auto& p:r.parameters){
    bool found=false;for(const auto& old:roots)found|=old.is_same(p);if(!found)roots.push_back(p);}}
  const auto unused=requests[2].content;
  requests.push_back(requests[0]);
  auto actual=col::source_columns(requests,Index{4}*1024*1024);std::vector<Tensor> expected;
  for(const auto& r:requests){const auto& p=r.parameters;
    auto a=p[0]+at::matmul(p[3],p[4]),b=p[1]+at::matmul(p[5],p[6]);
    auto z=r.content.scalar_type()==at::kHalf?r.content.to(at::kFloat):r.content;
    auto u=(z/at::sqrt(r.epsilon+z.square().mean(-1,true))).to(dtype);
    auto h=at::matmul(u,a.t())+p[2];expected.push_back(r.content+r.alpha*at::matmul(h*at::sigmoid(h),b.t()));}
  const double rtol=dtype==at::kHalf?2e-2:dtype==at::kFloat?1e-5:1e-8;
  const double atol=dtype==at::kHalf?1e-3:dtype==at::kFloat?1e-6:1e-10;
  for(size_t i=0;i<actual.size();++i)close_value(actual[i],expected[i],rtol,atol);
  for(double factor:{1.,0.}){
    auto loss=[&](const std::vector<Tensor>& v){return factor*(v[0].square().sum()+.3*v[1].sum()+.4*v[3].sum());};
    auto ga=torch::autograd::grad({loss(actual)},roots,{},true,false,true),ge=torch::autograd::grad({loss(expected)},roots,{},true,false,true);
    for(size_t i=0;i<roots.size();++i){close_value(ga[i],ge[i],rtol,atol);if(roots[i].is_same(unused))require(!ga[i].defined(),"unused source column connected");}
  }
  {at::NoGradGuard guard;requests[0].parameters[3].add_(.01);}
  bool rejected=false;try{torch::autograd::grad({actual[0].sum()},roots,{},false,false,true);}catch(const std::exception&){rejected=true;}
  require(rejected,"source column original version was not guarded");
}
void check_prepare_groups(at::Device device=at::kCPU,at::ScalarType dtype=at::kDouble){
  auto g=weighted_check::graph("fc","kv");auto m=weighted_check::model(g,"kv",device,dtype);
  const Index base=Index{1}<<54;
  col::Layout layout{{0,2,4},{base+10,base+13,base+19,base+22}};
  std::vector<col::KVPrepareRequest> requests;
  std::vector<Tensor> roots;for(const auto& owner:m.parameters().owners())roots.push_back(owner.value);
  const auto parameter_count=roots.size();
  for(Index i=0;i<5;++i){const auto& w=m.nodes[i];auto c=std::get<weighted::KVConfig>(weighted::node_binding(w).memory);
    auto mass=at::tensor({.3,0.,.6,.2},at::kDouble).to(device,dtype);
    if(i==3)mass=at::zeros_like(mass);else if(i==4)mass=mass+.1;
    mass.set_requires_grad(true);
    auto x=(at::arange(8,at::kDouble)*.03+.1+.02*i).reshape({4,2}).to(device,dtype).set_requires_grad(true);
    auto left=col::KVState::from_states({w.kernel->initial(w),w.kernel->initial(w)});
    requests.push_back({&w,c,layout,mass,x,left});roots.insert(roots.end(),{mass,x});
  }
  auto grouped=col::prepare_many(requests,Index{4}*1024*1024),single=col::prepare_many(requests,1);
  const double rtol=dtype==at::kHalf?2e-2:dtype==at::kFloat?1e-5:1e-8;
  const double atol=dtype==at::kHalf?1e-3:dtype==at::kFloat?1e-6:1e-10;
  for(size_t i=0;i<requests.size();++i){
    require(grouped[i].bank_layout.times==single[i].bank_layout.times,"group KV bank clocks changed");
    require(grouped[i].terminal.last_times==single[i].terminal.last_times&&
            grouped[i].terminal.observations==single[i].terminal.observations,"group KV raw metadata changed");
    for(const auto& [name,t]:grouped[i].bank)close_value(t,single[i].bank.at(name),rtol,atol);
    for(const auto& [name,t]:grouped[i].terminal.fields)close_value(t,single[i].terminal.fields.at(name),rtol,atol);
  }
  require(grouped[3].terminal.fields.at("keys").is_same(requests[3].left.fields.at("keys")),"zero-write group changed raw state");
  for(double factor:{1.,0.}){
    auto a=factor*(grouped[0].bank.at("keys").square().sum()+grouped[1].bank.at("values").square().sum());
    auto b=factor*(single[0].bank.at("keys").square().sum()+single[1].bank.at("values").square().sum());
    auto ga=torch::autograd::grad({a},roots,{},true,false,true),gb=torch::autograd::grad({b},roots,{},true,false,true);
    for(size_t i=0;i<roots.size();++i)close_value(ga[i],gb[i],rtol,atol);
    require(!ga[parameter_count+5].defined(),"unused grouped content acquired a connected-zero gradient");
  }
  auto next=requests[0];next.left=grouped[0].terminal;next.layout={{0,1,2},{base+100,base+100}};
  next.mass=at::zeros({2},next.mass.options());next.content=next.content.slice(0,0,2);
  auto silent=col::prepare_many({next},Index{4}*1024*1024).front();
  require(silent.terminal.layout.times==next.left.layout.times&&silent.terminal.fields.at("keys").is_same(next.left.fields.at("keys")),
          "silent grouped window expired raw history");
}
void check_query_groups(at::Device device=at::kCPU,at::ScalarType dtype=at::kDouble){
  at::set_num_threads(1);
  for(bool positional:{false,true}){
    auto g=weighted_check::graph("fc","kv");auto m=weighted_check::model(g,"kv",device,dtype);
    for(auto& w:m.nodes){auto b=weighted::node_binding(w);auto c=std::get<weighted::KVConfig>(b.memory);
      c.memory_read="qry";c.window=17;
      if(positional){c.read_frequencies={.13};c.full_frequencies={{.17}};c.distance_slopes={.03};c.position_scale=4.;}
      for(const auto* name:{"kernel.read_q","kernel.read_k"})
        w.extra[name]=w.extra.at("kernel.summary").detach().clone().set_requires_grad(true);
      w.extra["kernel.read_v"]=w.extra.at("kernel.read_k");
      w.extra["kernel.summary"]=m.nodes[0].extra.at("kernel.summary");
      w.extra["full_program.content.query"]=m.nodes[0].extra.at("full_program.content.query");
      weighted::configure_node(w,c,b.regional,b.sources,b.aggregate_alpha);
    }
    std::vector<Tensor> roots;for(const auto& p:m.parameters().owners())roots.push_back(p.value);
    const Index base=Index{1}<<54;const col::Layout layout{{0,2,4},{base+10,base+13,base+11,base+22}};
    std::vector<col::KVReadRequest> reads;std::vector<col::KVFullRequest> fulls;std::vector<size_t> contents;
    for(Index i=0;i<5;++i){const auto& w=m.nodes[i];const auto binding=weighted::node_binding(w);
      const auto c=std::get<weighted::KVConfig>(binding.memory);std::vector<State> old;
      for(Index b=0;b<2;++b){auto s=w.kernel->initial(w);s.last_time=base+b+1;s.observations=1;
        s.slots["times"]=at::tensor(std::vector<Index>{s.last_time},at::kLong).to(device);
        for(const auto* name:{"keys","values","summaries","weights"}){
          auto shape=std::string(name)=="weights"?std::vector<Index>{1}:std::vector<Index>{1,2};
          auto value=at::full(shape,.2+.03*i+.01*b,w.bias.options()).set_requires_grad(true);
          s.slots[name]=value;roots.push_back(value);
        }
        old.push_back(std::move(s));
      }
      auto mass=at::tensor({.3,0.,.6,.2},at::kDouble).to(device,dtype).set_requires_grad(true);
      auto x=(at::arange(8,at::kDouble)*.03+.1+.02*i).reshape({4,2}).to(device,dtype).set_requires_grad(true);
      roots.push_back(mass);contents.push_back(roots.size());roots.push_back(x);
      auto left=col::KVState::from_states(old);auto window=col::prepare(w,c,layout,mass,x,left);
      reads.push_back({&w,c,binding.regional,window,"proposal",left});
      auto controls=at::tensor({.2,0.,.4,.3},at::kDouble).to(device,dtype);
      if(i==3)controls=at::zeros_like(controls);
      controls.set_requires_grad(true);roots.push_back(controls);fulls.push_back({&w,c,window,controls});
    }
    const double rtol=dtype==at::kHalf?2e-2:dtype==at::kFloat?1e-5:1e-8;
    const double atol=dtype==at::kHalf?1e-3:dtype==at::kFloat?1e-6:1e-10;
    auto close=[&](const Tensor& a,const Tensor& b){close_value(a,b,rtol,atol);};
    auto vjp=[&](const Tensor& a,const Tensor& b){for(double factor:{1.,0.}){
      auto ga=torch::autograd::grad({factor*a.sum()},roots,{},true,false,true);
      auto gb=torch::autograd::grad({factor*b.sum()},roots,{},true,false,true);
      for(size_t j=0;j<roots.size();++j)close(ga[j],gb[j]);
      require(!ga[contents[2]].defined(),"unused grouped query content acquired a zero gradient");
    }};
    for(const auto* mode:{"content","old","proposal"}){
      for(auto& request:reads)request.mode=mode;
      auto actual=col::read_many(reads,Index{4}*1024*1024),single=col::read_many(reads,1);
      for(size_t i=0;i<reads.size();++i)for(const auto& [name,value]:actual[i])close(value,single[i].at(name));
      for(const auto& [name,value]:actual[0])if(value.requires_grad())vjp(value,single[0].at(name));
      vjp(actual[0].at("score")+actual[1].at("score"),single[0].at("score")+single[1].at("score"));
    }
    auto actual=col::full_many(fulls,Index{4}*1024*1024),single=col::full_many(fulls,1);
    for(size_t i=0;i<fulls.size();++i){require(actual[i].rows==single[i].rows,"grouped Full support changed");
      close(actual[i].strength,single[i].strength);close(actual[i].content,single[i].content);}
    require(actual[3].rows.empty(),"inactive Full was evaluated");
    vjp(actual[0].content,single[0].content);vjp(actual[0].strength,single[0].strength);
    vjp(actual[0].content+actual[1].content,single[0].content+single[1].content);
  }
}
void check(const std::string& family,bool positional=false,bool scoring=false,
           at::Device device=at::kCPU,at::ScalarType dtype=at::kDouble,const std::string& preset="native",
           std::optional<at::Device> peer=std::nullopt){
  at::set_num_threads(1);
  const bool half=dtype==at::kHalf,fp32=scoring||dtype==at::kFloat;
  auto close=[&](const Tensor& a,const Tensor& b){close_value(a,b,half?2e-2:fp32?1e-5:1e-8,half?1e-3:fp32?1e-6:1e-10);};
  auto g=weighted_check::graph("fc",family);auto m=weighted_check::model(g,family,device,dtype);
  if(positional)for(auto& node:m.nodes){auto b=weighted::node_binding(node);auto c=std::get<weighted::KVConfig>(b.memory);
    c.memory_read="qry";c.read_frequencies={.13};c.full_frequencies={{.17}};c.distance_slopes={.03};c.position_scale=4.;
    for(const auto* name:{"kernel.read_q","kernel.read_k","kernel.read_v"})
      node.extra[name]=node.extra.at("kernel.summary").detach().clone().set_requires_grad(true);
    weighted::configure_node(node,c,b.regional,b.sources,b.aggregate_alpha);}
  if(peer){std::vector<at::Device> owners;
    for(const auto& n:g.nodes)owners.push_back(n.region%2?*peer:device);
    m=place_payloads(g,m,owners);
  }
  ExecutionPlacement placement;placement.preset=preset;placement.scoring_dtype=scoring||half?"float32":"payload";
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
      const auto target=m.nodes[v].bias.device();
      for(const auto& [field,t]:n.descriptors)require(t.device()==resolve_placement(placement,target).read,"column Read location mismatch");
      require(n.controls.device()==target,"column control payload owner mismatch");
    }
    for(const auto& [e,p]:actual.messages)require(p.content.device()==m.nodes[g.edges[e].target].bias.device(),"column message target owner mismatch");
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
    for(auto& t:aterms)t=t.to(device);for(auto& t:eterms)t=t.to(device);
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
  if(argc==1){check_source_groups();check_prepare_groups();check_query_groups();check("kv");check("kv",true);for(const auto& f:{"affine","delta","gated_delta","ssd"})check(f);
    check("kv",false,true);check("ssd",false,true);
  }else{
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    args.allow_npu_float16=true;
    auto device=portable_torch::resolve_device(args);
    check_source_groups(device,args.dtype);
    check_prepare_groups(device,args.dtype);
    check_query_groups(device,args.dtype);
    std::optional<at::Device> peer;
    if(auto selected=std::getenv("TIDE_COLUMN_PEER_DEVICE")){auto peer_options=args;
      peer_options.device_spec=selected;peer=portable_torch::resolve_device(peer_options);
      require(*peer!=device&&peer->type()==device.type(),"peer gate requires a distinct same-backend device");}
    require(args.dtype==at::kHalf||args.dtype==at::kFloat||args.dtype==at::kDouble,"column gate requires FP16/FP32/FP64");
    for(const auto& preset:{"mixed-a","mixed-b","mixed-c"})for(const auto& family:{"kv","affine","delta","gated_delta","ssd"}){
      check(family,false,false,device,args.dtype,device.is_cpu()?"cpu":preset,peer);
      std::cout<<"column case passed: "<<family<<" "<<preset<<'\n';
    }
    if(args.dtype==at::kHalf){check("kv",true,false,device,args.dtype,device.is_cpu()?"cpu":"mixed-c",peer);
      std::cout<<"column case passed: kv positional half\n";}
    portable_torch::synchronize(device);
    if(peer)portable_torch::synchronize(*peer);
  }
  runtime.close();
  std::cout<<"column standalone: independent streaming, raw state, VJP, two SGD windows passed\n";return 0;}
  catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
