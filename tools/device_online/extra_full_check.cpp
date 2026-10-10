#include "full_vjp_fixture.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide::device_online;using at::Tensor;using I=int64_t;
struct Fixture {test::FullFixture base;int kind;std::vector<Tensor> banks;bool anchor=false;};
Fixture fixture(I d,int kind,int mode,at::ScalarType payload,bool anchor=false) {
  Fixture f{test::full_fixture(d,mode,false),kind,{}};auto& t=f.base.tape;
  f.anchor=anchor;
  const float nan=std::numeric_limits<float>::quiet_NaN();
  for(I row=0;row<t.values.size(0);++row) {
    const auto n=t.metadata[row][1].item<I>();
    if(n==2){f.base.connected[row].fill_(false);f.base.gradient[row].fill_(nan);}
    if(n==0&&t.metadata[row][3].item<I>())t.values[row].narrow(0,3*d,d).copy_(at::arange(d,at::kFloat).remainder(13)*.03-.3+row*.01);
    if(payload==at::kHalf&&t.metadata[row][3].item<I>())t.values[row].narrow(0,0,d).fill_(32768.f);
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
  if(payload==at::kHalf) {
    for(auto* bank:{&t.extra.lh_weights,&t.extra.lh_biases,&t.extra.gate,&t.extra.up,&t.extra.down})
      if(bank->defined())*bank=bank->to(at::kHalf);
    t.values=t.values.to(at::kHalf).to(at::kFloat);
    if(anchor) {
      for(I row=0;row<t.values.size(0);++row)if(t.metadata[row][1].item<I>()==0&&f.base.connected[row].item<bool>()) {
        auto x=kind==10?at::full({d},3.123046875f):at::arange(1,d+1,at::kFloat)*.125f;
        t.values[row].narrow(0,3*d,d).copy_(x);
        f.base.gradient[row].copy_(kind==10?at::ones_like(x):x);
      }
      if(kind<10){t.extra.lh_weights[0].fill_(1);t.extra.lh_biases[0].zero_();}
      else{t.extra.gate[0].fill_(1.234375);t.extra.up[0].fill_(.8125);t.extra.down[0].fill_(2.03125);}
    }
    f.banks=kind<10?std::vector<Tensor>{t.extra.lh_weights,t.extra.lh_biases}:
      std::vector<Tensor>{t.extra.gate,t.extra.up,t.extra.down};
  }
  return f;
}
FullTape upload(FullTape t,at::Device device) {
  for(auto* x:{&t.metadata,&t.values,&t.count,&t.kinds,&t.extra.lh_kinds,&t.extra.lh_weights,&t.extra.lh_biases,
    &t.extra.swiglu_kinds,&t.extra.swiglu_mapping,&t.extra.gate,&t.extra.up,&t.extra.down})if(x->defined())*x=x->to(device);
  return t;
}
Tensor rounded(const Tensor& x) {return x+(x.detach().to(at::kHalf).to(x.scalar_type())-x.detach());}
Tensor half_matmul(const Tensor& a,const Tensor& b) {
  auto y=at::matmul(a,b),saved=at::matmul(a.detach().to(at::kHalf),b.detach().to(at::kHalf)).to(y.scalar_type());
  return y+(saved-y.detach());
}
Tensor half_silu(const Tensor& x) {
  auto y=at::silu(x),saved=at::silu(x.detach().to(at::kHalf)).to(y.scalar_type());
  return y+(saved-y.detach());
}
std::vector<Tensor> reference(const Fixture& f,at::ScalarType dtype,bool quantized=true) {
  at::AutoGradMode enabled(true);const auto& t=f.base.tape;const auto capacity=t.values.size(0),d=t.width;
  const bool half=quantized&&f.banks[0].scalar_type()==at::kHalf;
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
        y=act==0?at::relu(c[i]):act==1?(half?half_silu(c[i]):at::silu(c[i])):c[i];
        if(half&&norm) {
          y=norm==1?at::rms_norm(y,{d},std::nullopt,1e-7):at::layer_norm(y,{d},std::nullopt,std::nullopt,1e-5);
          y=rounded(rounded(y)*parameters[0]);
          if(norm==2)y=rounded(y+parameters[owners]);
        }else if(norm==1)y=at::rms_norm(y,{d},parameters[0],1e-7);
        else if(norm==2)y=at::layer_norm(y,{d},parameters[0],parameters[owners],1e-5);
      }else if(half) {
        auto a=half_matmul(c[i],parameters[0]),b=half_matmul(c[i],parameters[owners]);
        y=rounded(y+half_matmul(rounded(half_silu(a)*b),parameters[2*owners]));
      }else y=y+at::matmul(at::silu(at::matmul(c[i],parameters[0]))*at::matmul(c[i],parameters[owners]),parameters[2*owners]);
    }
    terms.push_back((y*f.base.gradient[i].to(dtype)).sum());
  }
  for(const auto& group:{h,c,parameters})leaves.insert(leaves.end(),group.begin(),group.end());
  std::vector<Tensor> reference(leaves.size());if(!terms.empty())reference=torch::autograd::grad({at::stack(terms).sum()},leaves,{},false,false,true);
  return reference;
}
void compare(const Fixture& f,const FullVjp& v,at::ScalarType dtype) {
  const auto& t=f.base.tape;const auto capacity=t.values.size(0);const I owners=f.kind<10?4:2;
  const auto expected=reference(f,dtype);const bool half=f.banks[0].scalar_type()==at::kHalf;
  auto same=[&](const Tensor& x,const Tensor& on,const Tensor& y,const char* name){
    test::full_same_precision(x,on,y,name,half&&!f.anchor);
  };
  auto dh=v.content.cpu(),dc=v.comparison.cpu(),hc=v.content_connected.cpu(),cc=v.comparison_connected.cpu(),pc=v.parameter_connected.cpu();
  for(I i=0;i<capacity;++i){
    try{same(dh[i],hc[i],expected[i],"extra Full content");same(dc[i],cc[i],expected[capacity+i],"extra Full comparison");}
    catch(...){std::cerr<<"row="<<i<<" oracle="<<dtype<<'\n';throw;}
  }
  std::vector<Tensor> grads=f.kind<10?std::vector<Tensor>{v.extra.lh_weights.cpu(),v.extra.lh_biases.cpu()}:
    std::vector<Tensor>{v.extra.gate.cpu(),v.extra.up.cpu(),v.extra.down.cpu()};
  I index=2*capacity;
  for(size_t bank=0;bank<grads.size();++bank)for(I owner=0;owner<owners;++owner) {
    const bool used=f.kind==10||((f.kind-1)%3>0&&(bank==0||(f.kind-1)%3==2));
    auto on=used?pc[f.kind==10?owner*2:owner]:at::zeros({},at::kBool);
    same(grads[bank][owner],on,expected[index++],"extra Full parameter");
  }
}
void run(at::Device device,I width,int kind,int mode,at::ScalarType payload,bool anchor=false) {
  at::NoGradGuard guard;auto f=fixture(width,kind,mode,payload,anchor);auto t=upload(f.base.tape,device);
  if(anchor) {
    // A wrong all-FP32 recomputation must be observably different, otherwise
    // this is not a useful regression for saved half activation/product values.
    auto exact=reference(f,at::kFloat),wrong=reference(f,at::kFloat,false);bool different=false;
    for(size_t i=0;i<exact.size();++i)if(exact[i].defined()&&!at::allclose(exact[i],wrong[i],1e-5,1e-6))different=true;
    if(!different)throw std::runtime_error("extended Full half anchor does not distinguish unrounded recomputation");
  }
  auto g=f.base.gradient.to(device),on=f.base.connected.to(device),error=at::zeros({1},g.options().dtype(at::kInt));
  DeviceProgram p(device);p.limit_workspace(64*1024*1024);auto result=append_full_vjp(p,t,g,on,error,mode==1?4:1,64*1024*1024);p.finish();
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
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("extra Full VJP requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(int kind=1;kind<=10;++kind)for(I width:{1,7,257})for(int mode:{0,1,3}) {
      try{run(device,width,kind,mode,args.dtype);++cases;}
      catch(...){std::cerr<<"extra Full kind="<<kind<<" width="<<width<<" mode="<<mode<<'\n';throw;}
    }
    int anchors=0;if(args.dtype==at::kHalf)for(int kind:{8,9,10}) {
      try{run(device,kind==10?1:7,kind,1,args.dtype,true);++anchors;}
      catch(...){std::cerr<<"extra Full half anchor kind="<<kind<<'\n';throw;}
    }
    std::cout<<"device-extra-full-vjp: passed cases="<<cases<<" replays="<<cases*3<<" rounding_anchors="<<anchors
      <<" CPU=FP32_FP64 poison=true None_zero=true\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
