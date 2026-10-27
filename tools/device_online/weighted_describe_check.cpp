#include "weighted_describe_formula.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <iostream>
namespace {
using namespace tide::device_online;using namespace tide::device_online::describe_check;using at::Tensor;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void close(const Tensor& a,const Tensor& b){require(a.sizes()==b.sizes(),"Describe result shape");if(!at::allclose(a.cpu(),b,2e-5,2e-6)){
  std::cerr<<a.cpu()<<" versus "<<b<<'\n';throw std::runtime_error("Describe numeric mismatch");}}
Tensor values(at::IntArrayRef shape,float shift){int64_t n=1;for(auto d:shape)n*=d;return (at::arange(n,at::kFloat)*.13+shift).sin().reshape(shape)*.2;}
tide::weighted::RegionalConfig config(int variant){
  const std::vector<std::string> modes{"regional_total","regional_total","regional_threshold","regional_total_zero","constrained","direct_attention","proportional","regional_threshold_hard"};
  tide::weighted::RegionalConfig c(modes.at(variant));c.hidden_width=3;c.gamma_max=2;
  if(variant==1||variant==3||variant==4||variant==6){c.local_score="match";c.alpha_match=variant==3?0.:.7;c.alpha_res=variant==3?0.:.5;}
  if(variant==1||variant==2||variant==4||variant==6||variant==7){c.region_score="att";c.region_key=variant==1||variant==6?"content":"content_stats";
    c.query_weight=c.regional()?"received":"mu";c.alpha_loc=.7;c.alpha_sel=.5;}
  if(variant==1||variant==7)c.region_evidence="grouped";
  if(variant==2||variant==3){c.region_control_mode="fixed";c.region_evidence.reset();if(c.total())c.total_fraction=.4;else c.eta=.2;}
  if(variant==4){c.send_gain="context";c.fixed_gain.reset();}
  if(variant==5){c.gain_logit=.2;c.fixed_gain.reset();}
  if(variant==6)c.fixed_gain=1.4;
  if(c.local_score=="match"||c.region_score=="att"||c.region_evidence=="grouped")c.selection_width=2;
  c.validate();return c;
}
Fields inputs(int pass){
  Fields x{{"content",values({R,D},.2+pass*.1)},{"short_memory",values({R,DS},.4)},{"long_memory",values({R,DS},.7)},
    {"mass",at::tensor({0.f,.2f,.8f,1.f,1.5f,0.f}).reshape({R,1})},
    {"short_total",at::tensor({0.f,.3f,1.f,2.f,3.f,0.f}).reshape({R,1})},{"long_total",at::tensor({0.f,.7f,2.f,4.f,5.f,0.f}).reshape({R,1})}};
  x["short_memory"][0].zero_();x["long_memory"][0].zero_();for(auto& [name,value]:x){value[5].fill_(NAN);if(!pass)value[4].fill_(NAN);}return x;
}
WeightedDescribeInput unpack(const Fields& x){return {x.at("mass").reshape({R}),x.at("content"),x.at("short_memory"),x.at("long_memory"),x.at("short_total").reshape({R}),x.at("long_total").reshape({R})};}
void compare(const Formula& ref,const std::map<std::string,WeightedRowAdjoint>& roots,const WeightedDescribeTape& t,const WeightedDescribeGradient& g){
  at::AutoGradMode enabled(true);std::vector<Tensor> outputs,bars;
  for(int r=0;r<R;++r)for(const auto& [name,expected]:ref.fields[r]){close(t.fields.at(name)[r],expected);const auto& root=roots.at(name);
    if(root.connected[r].item<bool>()&&expected.requires_grad()){outputs.push_back(expected);bars.push_back(root.value[r]);}}
  auto derivatives=outputs.empty()?std::vector<Tensor>(ref.leaves.size()):torch::autograd::grad(outputs,ref.leaves,bars,true,false,true);
  for(size_t j=0;j<ref.leaves.size();++j){const auto& label=ref.labels[j];const auto& actual=(label.parameter?g.parameters:g.inputs).at(label.name);const auto& e=derivatives[j];
    if(actual.connected[label.row].cpu().item<bool>()!=e.defined()){std::cerr<<label.name<<" row="<<label.row<<" parameter="<<label.parameter<<'\n';throw std::runtime_error("Describe None/zero differs");}
    if(e.defined())try{close(actual.value[label.row],e);}catch(...){std::cerr<<label.name<<" row="<<label.row<<'\n';throw;}
    else require(actual.value[label.row].cpu().eq(0).all().item<bool>(),"absent Describe derivative nonzero");}
}
void check(at::Device device,int variant){
  at::NoGradGuard ng;const auto c=config(variant);auto x=inputs(0);Fields device_input;for(const auto& [name,value]:x)device_input[name]=value.to(device);
  auto owners=at::zeros({R},at::TensorOptions().device(device).dtype(at::kLong)),error=at::zeros({1},owners.options().dtype(at::kInt));WeightedDescribeParameters weights;
  for(const auto& [name,shape]:weighted_describe_parameter_shapes(B,D,DS,c))weights[name]=at::zeros(shape,owners.options().dtype(at::kFloat));
  DeviceProgram p(device);p.limit_workspace(128*1024*1024);auto t=append_weighted_describe(p,unpack(device_input),owners,weights,c,.13,error);
  std::map<std::string,WeightedRowAdjoint> roots;for(const auto& [name,value]:t.fields)roots[name]={at::zeros_like(value),at::zeros({R},owners.options().dtype(at::kBool))};
  auto g=append_weighted_describe_vjp(p,t,roots,error);p.finish();
  for(int pass=0;pass<2;++pass){auto x=inputs(pass);auto ids=at::tensor(std::vector<int64_t>{0,1,pass?2:0,0,pass?1:-1,-1},at::kLong);WeightedDescribeParameters parameters;
    int j=0;for(const auto& [name,value]:weights){auto data=values(value.sizes(),.1+pass*.1+j++*.017);if(!pass)data[2].fill_(NAN);parameters[name]=data;}
    auto ref=formula(x,ids,parameters,c);for(const auto& [name,value]:x)device_input[name].copy_(value);owners.copy_(ids);for(const auto& [name,value]:parameters)weights[name].copy_(value);
    const auto count=roots.size();for(size_t test=0;test<=count;++test){std::map<std::string,WeightedRowAdjoint> host;int k=0;
      for(const auto& [name,root]:roots){auto flags=at::zeros({R},at::kBool),bar=at::full(root.value.sizes(),NAN,at::kFloat);
        for(int r=0;r<R;++r)if(test==count?(r+k)%3!=0:test==size_t(k)){flags[r]=true;if(r!=5)bar[r].copy_(r==3?at::zeros_like(bar[r]):values(bar[r].sizes(),.8));}
        root.value.copy_(bar);root.connected.copy_(flags);host[name]={bar,flags};++k;}
      portable_torch::synchronize(device);p.run();require(!error.cpu().item<int>(),"Describe rejected valid fields");compare(ref,host,t,g);
      if(pass&&test==count){DeviceProgram reverse(device);auto saved=append_weighted_describe_vjp(reverse,t,roots,error);reverse.finish();
        for(auto& [name,value]:device_input)value.fill_(NAN);owners.fill_(-99);for(auto& [name,value]:weights)value.fill_(NAN);
        portable_torch::synchronize(device);reverse.run();require(!error.cpu().item<int>(),"Describe saved reverse reread source");compare(ref,host,t,saved);reverse.close();}
    }
  }
  portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()!=0,"Describe invalid source accepted");p.close();
  DeviceProgram refusal(device);bool denied=false;try{append_weighted_describe(refusal,unpack(device_input),owners,weights,c,.13,error,1);}catch(const std::invalid_argument&){denied=true;}
  require(denied,"Describe scratch budget omitted");refusal.close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("Describe requires explicit accelerator/FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    for(int variant=0;variant<8;++variant){try{check(device,variant);}catch(...){std::cerr<<"Describe variant="<<variant<<'\n';throw;}}
    std::cout<<"weighted-describe: passed configurations=8 all_independent_fields=true all_input_parameter_vjp=true saved_versions=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
