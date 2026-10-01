#include "attention_vjp.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

namespace {
using namespace tide::device_online;using I=int64_t;using at::Tensor;
struct Geometry {I heads,kv_heads,width,keys,tile;};
AttentionVjpInput fixture(Geometry g,int replay) {
  const float nan=std::numeric_limits<float>::quiet_NaN();const I rows=3;
  auto make=[](std::vector<I> shape,I shift) {
    I n=1;for(auto d:shape)n*=d;
    return ((at::arange(n,at::kLong)*7+shift).remainder(31).to(at::kFloat)/32.-.375).reshape(shape);
  };
  AttentionVjpInput x{make({rows,g.heads,g.width},replay),make({rows,g.kv_heads,g.keys,g.width},2+replay),
    make({rows,g.kv_heads,g.keys,g.width},5+replay),make({rows,g.keys},9+replay),
    at::tensor(replay==0?std::vector<I>{g.keys,std::min<I>(2,g.keys),g.keys}:
      replay==1?std::vector<I>{1,0,std::min<I>(3,g.keys)}:std::vector<I>{0,0,0},at::kLong),
    make({rows,g.heads,g.width},11+replay),at::zeros({rows},at::kBool)};
  if(replay<2){x.connected[0].fill_(true);x.connected[replay==0?1:2].fill_(true);x.cotangent[replay==0?1:2].zero_();}
  for(I row=0;row<rows;++row) {
    const I length=x.lengths[row].item<I>();
    if(!x.connected[row].item<bool>()) {
      x.query[row].fill_(nan);x.key[row].fill_(nan);x.value[row].fill_(nan);x.bias[row].fill_(nan);x.cotangent[row].fill_(nan);
    }else {
      x.key[row].narrow(1,length,g.keys-length).fill_(nan);x.value[row].narrow(1,length,g.keys-length).fill_(nan);
      x.bias[row].narrow(0,length,g.keys-length).fill_(nan);
      if(length>4)x.bias[row].narrow(0,0,2).fill_(-std::numeric_limits<float>::infinity());
    }
  }
  return x;
}
void compare(const AttentionVjpInput& x,const AttentionVjp& actual,Geometry g,at::ScalarType dtype) {
  at::AutoGradMode grad(true);
  std::vector<Tensor> observed{actual.query.cpu(),actual.key.cpu(),actual.value.cpu(),actual.bias.cpu()};
  if(!at::equal(actual.connected.cpu(),x.connected))throw std::runtime_error("attention None/zero connectivity mismatch");
  for(I row=0;row<3;++row) {
    if(!x.connected[row].item<bool>()) {
      for(const auto& v:observed)if(!at::isfinite(v[row]).all().item<bool>()||v[row].count_nonzero().item<I>())
        throw std::runtime_error("absent poisoned attention root produced values");
      continue;
    }
    std::vector<Tensor> leaves{x.query[row].to(dtype).clone().set_requires_grad(true),x.key[row].to(dtype).clone().set_requires_grad(true),
      x.value[row].to(dtype).clone().set_requires_grad(true),x.bias[row].to(dtype).clone().set_requires_grad(true)};
    auto heads=at::arange(g.heads,at::kLong)/(g.heads/g.kv_heads);heads=heads.to(at::kLong);
    const I size=x.lengths[row].item<I>();
    auto key=leaves[1].narrow(1,0,size).index_select(0,heads),value=leaves[2].narrow(1,0,size).index_select(0,heads);
    auto logits=at::matmul(leaves[0].unsqueeze(1),key.transpose(1,2))/std::sqrt(double(g.width));
    auto probability=at::softmax(logits+leaves[3].narrow(0,0,size),-1);
    auto output=at::matmul(probability,value).squeeze(1);
    auto expected=torch::autograd::grad({output},leaves,{x.cotangent[row].to(dtype)},false,false,true);
    for(size_t j=0;j<expected.size();++j) {
      auto got=observed[j][row].to(at::kDouble),ref=expected[j].to(at::kDouble);
      if(!expected[j].defined()||!at::isfinite(got).all().item<bool>()||!at::allclose(got,ref,2e-5,2e-6))
        throw std::runtime_error("attention packed VJP mismatch row="+std::to_string(row)+" root="+std::to_string(j)
          +" max="+std::to_string((got-ref).abs().max().item<double>()));
    }
  }
}
void check(at::Device device,Geometry g) {
  at::NoGradGuard guard;auto x=fixture(g,0);
  AttentionVjpInput input{x.query.to(device),x.key.to(device),x.value.to(device),x.bias.to(device),x.lengths.to(device),x.cotangent.to(device),x.connected.to(device)};
  auto error=at::zeros({1},input.query.options().dtype(at::kInt));
  CannProgram program(device);program.limit_workspace(64*1024*1024);
  auto out=append_attention_vjp(program,input,error,1./std::sqrt(double(g.width)),g.tile,64*1024*1024);program.finish();
  for(int replay=0;replay<3;++replay) {
    x=fixture(g,replay);
    input.query.copy_(x.query);input.key.copy_(x.key);input.value.copy_(x.value);input.bias.copy_(x.bias);
    input.lengths.copy_(x.lengths);input.cotangent.copy_(x.cotangent);input.connected.copy_(x.connected);
    portable_torch::synchronize(device);program.run();
    if(error.cpu().item<int>())throw std::runtime_error("valid reverse attention refused");
    compare(x,out,g,at::kFloat);compare(x,out,g,at::kDouble);
    if(replay==2&&out.tiles.cpu().item<I>()!=0)throw std::runtime_error("empty replay retained attention work");
  }
  input.lengths[0].fill_(g.keys+1);portable_torch::synchronize(device);program.run();
  if(error.cpu().item<int>()!=2)throw std::runtime_error("invalid attention length accepted");
  program.close();bool refused=false;
  try{CannProgram bad(device);append_attention_vjp(bad,input,error,1.,g.tile,1);}
  catch(const std::invalid_argument&){refused=true;}
  if(!refused)throw std::runtime_error("attention reverse memory budget ignored");
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("attention VJP gate requires explicit NPU FP32");
    const auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    for(auto g:std::vector<Geometry>{{4,4,3,7,2},{4,2,8,19,7},{6,2,17,257,64},{1,1,257,3,3}}) {
      check(device,g);std::cout<<"PASS attention VJP H="<<g.heads<<" KV="<<g.kv_heads<<" D="<<g.width<<" K="<<g.keys<<" tile="<<g.tile<<'\n'<<std::flush;
    }
    std::cout<<"device-attention-vjp: passed configurations=4 replays=12 CPU=FP32_FP64 GQA=true None_zero=true poison=true\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
