#include "training_test.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <cmath>

namespace {
using namespace tide;using namespace tide::device_online;
test::Fixture fixture(int kind,Index width=3) {
  auto f=test::retained_fixture(kind%2?0:3,0,width);
  const std::vector<std::string> act{"relu","silu","identity"},norm{"identity","rms","layer"};
  Tensor shared_gate,shared_norm;
  for(size_t n=0;n<f.graph.nodes.size();++n) {
    auto& node=f.graph.nodes[n];auto& w=f.model.nodes[n];if(node.identity)continue;
    const auto which=kind==10?int(n%3==0?9:n%3==1?5:-1):kind;
    if(which<0)continue; // Mixed tanh owner remains independently active.
    if(which==9) {
      node.full="swiglu";auto eye=at::eye(width,at::kFloat);
      if(!shared_gate.defined())shared_gate=at::cat({eye*.25f,eye*-.125f},1);
      w.extra["ffn_gate"]=shared_gate;w.extra["ffn_up"]=shared_gate;
      w.extra["ffn_down"]=at::cat({eye*.375f,eye*.125f},0);
    } else {
      node.full="lh-"+act[which/3]+"-"+norm[which%3]+"-v1";
      if(which%3) {
        if(!shared_norm.defined())shared_norm=at::arange(width,at::kFloat).remainder(5)*.0625f+.75f;
        w.extra["lh_norm_weight"]=shared_norm;
      }
      if(which%3==2)w.extra["lh_norm_bias"]=at::arange(width,at::kFloat).remainder(3)*.03125f;
    }
  }
  f.graph.compile();f.initial.identity=f.graph.identity;return f;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    int filter=-1;Index width=3;std::string reference="both";double epsilon=1e-5;bool numerics_only=false;
    std::string control_check="strict";bool control_selftest=false;
    std::vector<char*> forwarded{argv[0]};
    for(int i=1;i<argc;++i) {
      const std::string option=argv[i];
      if(option.rfind("--case=",0)==0)filter=std::stoi(option.substr(7));
      else if(option.rfind("--reference=",0)==0)reference=option.substr(12);
      else if(option.rfind("--adam-eps=",0)==0)epsilon=std::stod(option.substr(11));
      else if(option.rfind("--width=",0)==0)width=std::stoll(option.substr(8));
      else if(option.rfind("--control-check=",0)==0)control_check=option.substr(16);
      else if(option=="--control-selftest")control_selftest=true;
      else if(option=="--numerics-only")numerics_only=true;
      else forwarded.push_back(argv[i]);
    }
    if(filter < -1||filter>10||!std::isfinite(epsilon)||epsilon<0||width<1||(filter<0&&width!=3)
        ||(reference!="both"&&reference!="float32"&&reference!="float64")||(control_check!="strict"&&control_check!="conditioned"))
      throw std::invalid_argument("invalid Full training diagnostic case/reference");
    auto args=portable_torch::parse_cli(forwarded.size(),forwarded.data(),true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);
      std::cout<<"  --case=0..10 --width=N --reference=both|float32|float64 --adam-eps=VALUE --numerics-only\n"
        <<"  --control-check=strict|conditioned --control-selftest\n";return 0;}
    if(control_selftest){at::set_num_threads(1);test::train_control_checks();runtime.close();return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("Full training requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    // A separate near-zero case checks VJP and the optimizer from the same
    // actual device gradient at eps1e-8. It does not claim trajectory parity.
    if(filter<0||numerics_only)for(bool prefill:{false,true})test::train_numerics(device,fixture(7),prefill);
    if(numerics_only){runtime.close();return 0;}
    int cases=0;
    for(int kind=0;kind<=10;++kind)for(bool prefill:{false,true})for(auto dtype:{at::kFloat,at::kDouble}) {
      if((filter>=0&&kind!=filter)||(reference=="float32"&&dtype!=at::kFloat)||(reference=="float64"&&dtype!=at::kDouble))continue;
      try {test::train_trajectory(device,fixture(kind,width),prefill,
        prefill?ResidentOptimizerKind::adamw:ResidentOptimizerKind::sgd,dtype,epsilon,control_check=="conditioned");++cases;}
      catch(...){std::cerr<<"Full training kind="<<kind<<" prefill="<<prefill<<" reference="<<dtype<<'\n';throw;}
    }
    if(filter<0)for(int kind:{5,9}) {
      try{test::train_trajectory(device,fixture(kind,257),true,ResidentOptimizerKind::adamw,at::kFloat,epsilon,control_check=="conditioned");++cases;}
      catch(...){std::cerr<<"Full training kind="<<kind<<" prefill=1 reference=Float width=257\n";throw;}
    }
    std::cout<<"resident-full-training: passed trajectories="<<cases<<" windows="<<cases*16<<" updates="<<cases*4
      <<" scope="<<(filter<0&&reference=="both"?"full":"diagnostic")<<" case="<<filter<<" reference="<<reference
      <<" adam_epsilon="<<epsilon<<" control_check="<<control_check<<" shared_owners=true checkpoint_resume=true\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
