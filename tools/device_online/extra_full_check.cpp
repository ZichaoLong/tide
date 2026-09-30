#include "full_vjp_fixture.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide::device_online;using at::Tensor;using I=int64_t;
struct Fixture {test::FullFixture base;int kind;std::vector<Tensor> banks;};
Fixture fixture(I d,int kind,int mode) {
  Fixture f{test::full_fixture(d,mode,false),kind,{}};auto& t=f.base.tape;
  const float nan=std::numeric_limits<float>::quiet_NaN();
  for(I row=0;row<t.values.size(0);++row) {
    const auto n=t.metadata[row][1].item<I>();
    if(n==2){f.base.connected[row].fill_(false);f.base.gradient[row].fill_(nan);}
    if(n==0&&t.metadata[row][3].item<I>())t.values[row].narrow(0,3*d,d).copy_(at::arange(d,at::kFloat).remainder(13)*.03-.3+row*.01);
  }
  if(kind<10) {
    auto w=at::ones({5,d},at::kFloat),b=at::zeros_like(w);
    w[0].copy_(at::arange(d,at::kFloat).remainder(5)*.0625+.75);b[0].fill_(.03125);w[2].fill_(nan);b[2].fill_(nan);w[4].zero_();
    t.extra.lh_kinds=at::tensor({kind,0,kind,0},at::kLong);t.extra.lh_groups={kind};t.extra.lh_weights=w;t.extra.lh_biases=b;
    f.banks={w,b};
  }else {
    auto gate=at::zeros({3,d,2*d},at::kFloat),up=at::zeros_like(gate),down=at::zeros({3,2*d,d},at::kFloat);
    auto eye=at::eye(d,at::kFloat);gate[0].copy_(at::cat({eye*.25,eye*-.125},1));up[0].copy_(at::cat({eye*.125,eye*.375},1));
    down[0].copy_(at::cat({eye*.375,eye*.25},0));gate[1].fill_(nan);up[1].fill_(nan);down[1].fill_(nan);
    t.extra.swiglu_kinds=at::tensor({1,0,1,0},at::kLong);t.extra.swiglu_mapping=at::tensor({0,2,1,2,2},at::kLong);
    t.extra.gate=gate;t.extra.up=up;t.extra.down=down;f.banks={gate,up,down};
  }
  return f;
}
FullTape upload(FullTape t,at::Device device) {
  for(auto* x:{&t.metadata,&t.values,&t.count,&t.kinds,&t.extra.lh_kinds,&t.extra.lh_weights,&t.extra.lh_biases,
    &t.extra.swiglu_kinds,&t.extra.swiglu_mapping,&t.extra.gate,&t.extra.up,&t.extra.down})if(x->defined())*x=x->to(device);
  return t;
}
void compare(const Fixture& f,const FullVjp& v,at::ScalarType dtype) {
  at::AutoGradMode enabled(true);const auto& t=f.base.tape;const auto capacity=t.values.size(0),d=t.width;
  std::vector<Tensor> h,c,parameters,leaves,terms;
  auto leaf=[&](const Tensor& x){return x.detach().to(dtype).clone().set_requires_grad(true);};
  for(I i=0;i<capacity;++i){h.push_back(leaf(t.values[i].narrow(0,0,d)));c.push_back(leaf(t.values[i].narrow(0,3*d,d)));}
  const I owners=f.kind<10?4:2;
  for(const auto& bank:f.banks)for(I i=0;i<owners;++i)parameters.push_back(leaf(bank[i]));
  for(I i=0;i<t.count.item<I>();++i)if(f.base.connected[i].item<bool>()) {
    const auto n=t.metadata[i][1].item<I>();Tensor y=h[i];
    if(n==0) {
      if(f.kind<10) {
        const auto act=(f.kind-1)/3,norm=(f.kind-1)%3;
        y=act==0?at::relu(c[i]):act==1?at::silu(c[i]):c[i];
        if(norm==1)y=at::rms_norm(y,{d},parameters[0],1e-7);
        else if(norm==2)y=at::layer_norm(y,{d},parameters[0],parameters[owners],1e-5);
      }else y=y+at::matmul(at::silu(at::matmul(c[i],parameters[0]))*at::matmul(c[i],parameters[owners]),parameters[2*owners]);
    }
    terms.push_back((y*f.base.gradient[i].to(dtype)).sum());
  }
  for(const auto& group:{h,c,parameters})leaves.insert(leaves.end(),group.begin(),group.end());
  std::vector<Tensor> reference(leaves.size());if(!terms.empty())reference=torch::autograd::grad({at::stack(terms).sum()},leaves,{},false,false,true);
  auto dh=v.content.cpu(),dc=v.comparison.cpu(),hc=v.content_connected.cpu(),cc=v.comparison_connected.cpu(),pc=v.parameter_connected.cpu();
  for(I i=0;i<capacity;++i){test::full_same(dh[i],hc[i],reference[i],"extra Full content");test::full_same(dc[i],cc[i],reference[capacity+i],"extra Full comparison");}
  std::vector<Tensor> grads=f.kind<10?std::vector<Tensor>{v.extra.lh_weights.cpu(),v.extra.lh_biases.cpu()}:
    std::vector<Tensor>{v.extra.gate.cpu(),v.extra.up.cpu(),v.extra.down.cpu()};
  I index=2*capacity;
  for(size_t bank=0;bank<grads.size();++bank)for(I owner=0;owner<owners;++owner) {
    const bool used=f.kind==10||((f.kind-1)%3>0&&(bank==0||(f.kind-1)%3==2));
    auto on=used?pc[f.kind==10?owner*2:owner]:at::zeros({},at::kBool);
    test::full_same(grads[bank][owner],on,reference[index++],"extra Full parameter");
  }
}
void run(at::Device device,I width,int kind,int mode) {
  at::NoGradGuard guard;auto f=fixture(width,kind,mode);auto t=upload(f.base.tape,device);
  auto g=f.base.gradient.to(device),on=f.base.connected.to(device),error=at::zeros({1},g.options().dtype(at::kInt));
  CannProgram p(device);p.limit_workspace(64*1024*1024);auto result=append_full_vjp(p,t,g,on,error,mode==1?4:1,64*1024*1024);p.finish();
  for(I count:{19,7,0}) {
    f.base.tape.count.fill_(count);t.count.copy_(f.base.tape.count);portable_torch::synchronize(device);p.run();
    if(error.cpu().item<int>())throw std::runtime_error("extended Full refused valid tape");
    compare(f,result,at::kFloat);compare(f,result,at::kDouble);
  }
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("extra Full VJP requires explicit NPU FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(int kind=1;kind<=10;++kind)for(I width:{1,7,257})for(int mode:{0,1,3}) {
      try{run(device,width,kind,mode);++cases;}
      catch(...){std::cerr<<"extra Full kind="<<kind<<" width="<<width<<" mode="<<mode<<'\n';throw;}
    }
    std::cout<<"device-extra-full-vjp: passed cases="<<cases<<" replays="<<cases*3<<" CPU=FP32_FP64 poison=true None_zero=true\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
