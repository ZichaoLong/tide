// Retained finite-cut chain rule against independent CPU Greedy/autograd.
// Three windows include an empty identity window and a detached left cut.
#include "weighted_window_check.h"
#include "weighted_window_bridge.h"
#include "../../cpp/test/weighted_graph_fixture.h"
#include "../../cpp/bench/streaming.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
using namespace tide;
using namespace tide::device_online;
namespace tide::device_online::window_check {
void leaf(at::Tensor& x){x=x.detach().clone().set_requires_grad(true);}
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void execute(DeviceProgram& p,const at::Tensor& error,at::Device device,const std::string& operation){
  p.finish();portable_torch::synchronize(device);p.run(60000);
  const auto code=error.cpu().item<int>();
  if(code)throw std::runtime_error("retained "+operation+" refused, device error="+std::to_string(code));p.close();
}
void run_retained(at::Device device,const std::string& topology,const std::string& read,const std::vector<at::Device>& node_devices){
  auto g=weighted_check::graph(topology,"heterogeneous");for(auto& r:g.regions)r.read_mode=read;g.compile();
  auto m=weighted_check::model(g,"heterogeneous",at::kCPU,at::kFloat);auto registry=m.parameters();
  auto inputs=weighted_check::inputs(g,at::kCPU,at::kFloat);for(auto& x:inputs)leaf(x.value);
  WeightedFlowLimits limits;limits.cache=16;limits.stages=32;limits.queue=64;limits.tensor_bytes=768*1024*1024;
  limits.operator_bytes=128*1024*1024;limits.timeout_ms=60000;limits.prefill=read!="old";
  Continuation initial;initial.identity=g.identity;initial.batch_size=2;at::NoGradGuard ng;
  WeightedFlow flow(g,m,initial,device,limits,node_devices);
  std::vector<External> prefix;for(const auto& x:inputs)if(x.time<2)prefix.push_back(x);
  initial=flow.advance(prefix,2).continuation;
  for(auto& a:initial.pending)leaf(a.value);
  for(auto& [key,s]:initial.states){leaf(s.value);for(auto& [name,x]:s.slots)if(x.scalar_type()==at::kFloat)leaf(x);}
  std::vector<std::vector<External>> xs(3);
  for(const auto& x:inputs)if(x.time>=2)xs[x.time<4?0:2].push_back(x);
  std::vector<Result> actual,reference;std::vector<std::shared_ptr<WeightedWindowSnapshot>> saved;
  const std::vector<int64_t> stops{4,4,10};
  for(size_t i=0;i<stops.size();++i){actual.push_back(flow.advance(xs[i],stops[i]));saved.push_back(WeightedWindowSnapshot::retain(flow,64*1024*1024));}
  flow.close();
  {at::AutoGradMode grad(true);auto q=initial;Options options;options.packed=false;
    for(size_t i=0;i<stops.size();++i){reference.push_back(Greedy(g,m,options).run(q,xs[i],stops[i],stops[i]));q=reference.back().continuation;}}
  std::vector<std::unique_ptr<WeightedWindowAdjoint>> reverse;std::vector<Records> records;
  for(size_t i=0;i<saved.size();++i){tide_bench::compare(actual[i],reference[i],true,at::kFloat);
    reverse.push_back(std::make_unique<WeightedWindowAdjoint>(saved[i]));records.push_back(reverse.back()->records());}
  auto error=at::zeros({1},at::TensorOptions().device(device).dtype(at::kInt));
  for(const std::string mode:{"none","full","final","pending","all"}){
    std::cout<<"weighted-retained phase "<<topology<<'/'<<read<<' '<<mode<<'\n'<<std::flush;
    std::vector<at::Tensor> outputs,bars,leaves;
    for(size_t i=0;i<saved.size();++i){seed(*reverse[i],records[i],actual[i],mode);at::AutoGradMode grad(true);
      auto roots=objective(reference[i],*reverse[i],records[i]);outputs.insert(outputs.end(),roots.first.begin(),roots.first.end());bars.insert(bars.end(),roots.second.begin(),roots.second.end());}
    ParameterVjp total;
    for(size_t i=saved.size();i>0;){--i;
      if(i+1<saved.size()){DeviceProgram bridge(device);bridge.limit_workspace(limits.operator_bytes);
        append_weighted_window_bridge(bridge,*saved[i],reverse[i]->roots(),*saved[i+1],*reverse[i+1],error,64*1024*1024);execute(bridge,error,device,"bridge at "+std::to_string(i));}
      reverse[i]->run();auto part=bind_weighted_window_parameters(reverse[i]->parameters(),registry);
      if(total.values.defined()){DeviceProgram reduce(device);reduce.limit_workspace(limits.operator_bytes);
        total=append_parameter_accumulate(reduce,total,part,error,64*1024*1024);execute(reduce,error,device,"parameter reduction at "+std::to_string(i));}else total=part;
    }
    for(const auto& owner:registry.owners())leaves.push_back(owner.value);
    for(const auto& window:xs)for(const auto& x:window)leaves.push_back(x.value);
    for(const auto& a:initial.pending)leaves.push_back(a.value);
    for(const auto& [key,s]:initial.states){leaves.push_back(s.value);for(const auto& [name,x]:s.slots)if(x.scalar_type()==at::kFloat)leaves.push_back(x);}
    std::vector<at::Tensor> expected(leaves.size());
    if(!outputs.empty()){at::AutoGradMode grad(true);expected=torch::autograd::grad(outputs,leaves,bars,true,false,true);}
    size_t cursor=0;auto values=total.values.cpu(),flags=total.connected.cpu();
    for(size_t j=0;j<total.owners.size();++j){const auto& owner=total.owners[j];auto x=total.offsets[j]<0?at::zeros_like(owner.value):values.narrow(0,total.offsets[j],owner.value.numel()).reshape(owner.value.sizes());
      check_gradient(x,flags[j].item<bool>(),expected.at(cursor++),owner.canonical);}
    auto atom=[&](size_t window,const Atom& a){const auto& boundary=reverse[window]->boundary();auto coords=boundary.coordinates.cpu(),present=boundary.present.cpu();int64_t found=-1;
      for(int64_t i=0;i<present.numel();++i)if(present[i].item<bool>()&&same_atom(coords[i],a)){require(found<0,"ambiguous retained boundary");found=i;}
      require(found>=0,"missing retained boundary");check_gradient(boundary.values[found].cpu(),boundary.connected[found].item<bool>(),expected.at(cursor++),"retained input/pending");};
    for(size_t i=0;i<xs.size();++i)for(const auto& x:xs[i])atom(i,{x.batch,g.inputs[x.port],x.time,0,x.port,x.position,x.value});
    for(const auto& a:initial.pending)atom(0,a);
    for(const auto& [key,s]:initial.states){const auto& layout=saved[0]->layouts[key.second];const auto& r=reverse[0]->states()[key.second];
      auto one=[&](const std::string& name,const at::Tensor& x){size_t j=0;while(layout.fields()[j].name!=name)++j;
        check_gradient(r.value[key.first].narrow(0,layout.fields()[j].offset,x.numel()).reshape(x.sizes()).cpu(),r.connected[key.first][j].item<bool>(),expected.at(cursor++),"retained left raw "+name);};
      one("value",s.value);for(const auto& [name,x]:s.slots)if(x.scalar_type()==at::kFloat)one(name,x);}
    require(cursor==expected.size(),"retained gradient roots incomplete");
  }
  bool refused=false;try{DeviceProgram p(device);append_weighted_window_bridge(p,*saved[0],reverse[0]->roots(),*saved[0],*reverse[0],error,64*1024*1024);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"nonconsecutive retained cuts accepted");
  for(auto& r:reverse)r->close();std::cout<<"weighted-retained case "<<topology<<'/'<<read<<" passed windows=3 modes=5\n"<<std::flush;
}
}
