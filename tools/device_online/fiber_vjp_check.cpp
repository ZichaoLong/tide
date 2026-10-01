#include "fiber_vjp.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <cmath>
#include <iostream>
#include <limits>

namespace {
using namespace tide::device_online;using I=int64_t;using at::Tensor;
struct Fixture {FiberVjpInput in;Tensor old_key,old_value,old_bias;};
Tensor make(std::vector<I> shape,I shift) {
  I n=1;for(auto d:shape)n*=d;
  return ((at::arange(n,at::kLong)*7+shift).remainder(31).to(at::kFloat)/128.-.125).reshape(shape);
}
Fixture fixture(I w,I heads,int pool,int mode,int replay) {
  const I b=3,s=4,k=11,d=w/heads,domain=6;const float nan=std::numeric_limits<float>::quiet_NaN();
  Fixture f;auto& x=f.in;
  x.rows=make({b,s,w},replay);x.rows[0][0].zero_();
  x.slots=at::tensor(std::vector<I>{0,2,3,5,0,2,3,5,0,2,3,5},at::kLong).reshape({b,s});
  x.counts=at::tensor(replay==0?std::vector<I>{4,2,1}:std::vector<I>{1,3,0},at::kLong);
  x.old_lengths=at::tensor(replay==0?std::vector<I>{3,0,2}:std::vector<I>{1,2,0},at::kLong);
  x.lengths=x.old_lengths+x.counts;x.ticks=at::tensor(std::vector<I>{3,1,7},at::kLong);
  x.qkv=make({b,w,3*w},2);x.qkv_bias=make({b,3*w},4);x.projection=make({b,w,w},6);
  x.pool_kinds=at::full({b},pool,at::kLong);x.pool_lengths=at::full({b},domain,at::kLong);
  x.pool_weights=make({b,domain},8);x.pool_weights[0][2].zero_();
  x.cotangent=make({b,w},11);x.connected=at::full({b},mode==1||mode>=5,at::kBool);
  x.key_root=make({b,heads,k,d},12);x.value_root=make({b,heads,k,d},13);x.bias_root=make({b,k},14);
  x.key_on=at::full({b},mode==2||mode>=5,at::kBool);x.value_on=at::full({b},mode==3||mode>=5,at::kBool);
  x.bias_on=at::full({b},mode==4||mode>=5,at::kBool);
  f.old_key=make({b,heads,k,d},15);f.old_value=make({b,heads,k,d},16);f.old_bias=make({b,k},17);
  x.key=at::full({b,heads,k,d},nan);x.value=at::full_like(x.key,nan);x.bias=at::full({b,k},nan);
  if(mode==6)for(auto a:{x.cotangent,x.key_root,x.value_root,x.bias_root})a.zero_();
  for(I row=0;row<b;++row) {
    const I count=x.counts[row].item<I>(),length=x.lengths[row].item<I>();
    if(!count)for(auto a:{x.connected,x.key_on,x.value_on,x.bias_on})a[row].fill_(false);
    x.rows[row].narrow(0,count,s-count).fill_(nan);
    for(auto pair:{std::make_pair(x.key_root,x.key_on),std::make_pair(x.value_root,x.value_on)})
      pair.first[row].narrow(1,pair.second[row].item<bool>()?length:0,k-(pair.second[row].item<bool>()?length:0)).fill_(nan);
    x.bias_root[row].narrow(0,x.bias_on[row].item<bool>()?length:0,k-(x.bias_on[row].item<bool>()?length:0)).fill_(nan);
    if(!x.connected[row].item<bool>())x.cotangent[row].fill_(nan);
  }
  return f;
}
std::vector<Tensor*> tensors(FiberVjpInput& x) {
  return {&x.rows,&x.slots,&x.counts,&x.key,&x.value,&x.bias,&x.lengths,&x.old_lengths,&x.ticks,&x.qkv,&x.qkv_bias,&x.projection,
    &x.pool_kinds,&x.pool_lengths,&x.pool_weights,&x.cotangent,&x.connected,&x.key_root,&x.value_root,&x.bias_root,&x.key_on,&x.value_on,&x.bias_on};
}
FiberVjpInput device_input(const Fixture& f,at::Device device) {
  auto x=f.in;for(auto* t:tensors(x))*t=t->to(device);
  const I w=x.rows.size(2),h=x.key.size(1),d=w/h;
  // Construct the candidate's own forward cache on NPU. The independent CPU
  // oracle below recomputes its cache; no oracle values/gradients are imported.
  for(I b=0;b<3;++b) {
    const I old=f.in.old_lengths[b].item<I>(),count=f.in.counts[b].item<I>();if(!count)continue;
    auto parts=(at::matmul(x.rows[b].narrow(0,0,count),x.qkv[b])+x.qkv_bias[b]).split(w,-1);
    auto key=at::cat({f.old_key[b].narrow(1,0,old).to(device),parts[1].reshape({count,h,d}).transpose(0,1)},1);
    auto value=at::cat({f.old_value[b].narrow(1,0,old).to(device),parts[2].reshape({count,h,d}).transpose(0,1)},1);
    auto bias=f.old_bias[b].narrow(0,0,old).to(device);
    if(old)for(I t=0;t<f.in.ticks[b].item<I>();++t)bias=bias-.01;
    x.key[b].narrow(1,0,old+count).copy_(key);x.value[b].narrow(1,0,old+count).copy_(value);
    x.bias[b].narrow(0,0,old+count).copy_(at::cat({bias,at::zeros({count},bias.options())}));
  }
  return x;
}
void same(const Tensor& actual,bool connected,const Tensor& expected,const std::string& name) {
  if(connected!=expected.defined())throw std::runtime_error("fiber VJP connection mismatch "+name);
  auto got=actual.to(at::kDouble);auto ref=expected.defined()?expected.to(at::kDouble):at::zeros_like(got);
  if(!at::isfinite(got).all().item<bool>()||!at::allclose(got,ref,2e-5,2e-6))
    throw std::runtime_error("fiber VJP numeric mismatch "+name+" max="+std::to_string(got.numel()?(got-ref).abs().max().item<double>():0.));
}
void compare(const Fixture& f,const FiberVjp& out,at::ScalarType dtype) {
  at::AutoGradMode grad(true);const auto& x=f.in;const I w=x.rows.size(2),h=x.key.size(1),d=w/h,capacity=x.key.size(2);
  auto input=out.rows.cpu(),icon=out.rows_connected.cpu(),key=out.key.cpu(),value=out.value.cpu(),bias=out.bias.cpu();
  auto cc=out.cache_connected.cpu(),pc=out.parameter_connected.cpu();
  std::vector<Tensor> observed{out.qkv.cpu(),out.qkv_bias.cpu(),out.projection.cpu(),out.projection_bias.cpu(),out.decay.cpu(),out.pool.cpu()};
  for(I b=0;b<3;++b) {
    const I count=x.counts[b].item<I>(),old=x.old_lengths[b].item<I>(),pool=x.pool_kinds[b].item<I>();
    std::vector<Tensor> source{x.rows[b].narrow(0,0,count),x.qkv[b],x.qkv_bias[b],x.projection[b],at::zeros({w},at::kFloat),
      at::full({},.01,at::kFloat),x.pool_weights[b],f.old_key[b].narrow(1,0,old),f.old_value[b].narrow(1,0,old),f.old_bias[b].narrow(0,0,old)};
    for(auto& a:source)a=a.to(dtype).clone().set_requires_grad(true);
    std::vector<Tensor> expected(source.size());
    if(count) {
      auto parts=(at::matmul(source[0],source[1])+source[2]).split(w,-1);
      auto q=parts[0].reshape({count,h,d}).transpose(0,1),k=at::cat({source[7],parts[1].reshape({count,h,d}).transpose(0,1)},1);
      auto v=at::cat({source[8],parts[2].reshape({count,h,d}).transpose(0,1)},1),log=source[9];
      if(old)for(I t=0;t<x.ticks[b].item<I>();++t)log=log-source[5];
      log=at::cat({log,at::zeros({count},log.options())});
      auto ys=at::matmul(at::softmax(at::matmul(q,k.transpose(1,2))/std::sqrt(double(d))+log,-1),v)
        .transpose(0,1).reshape({count,w});
      auto slots=x.slots[b].narrow(0,0,count);Tensor coefficients;
      if(pool==0)coefficients=at::ones({count},ys.options());
      else if(pool==1)coefficients=at::full({count},1./count,ys.options());
      else if(pool==2)coefficients=source[6].index_select(0,slots);
      else if(pool==3)coefficients=at::softmax(source[6].index_select(0,slots),0);
      else coefficients=at::softmax(source[6],0).index_select(0,slots);
      auto proposal=at::matmul((ys*coefficients.unsqueeze(1)).sum(0),source[3])+source[4];
      std::vector<Tensor> terms;
      if(x.connected[b].item<bool>())terms.push_back((proposal*x.cotangent[b].to(dtype)).sum());
      if(x.key_on[b].item<bool>())terms.push_back((k*x.key_root[b].narrow(1,0,old+count).to(dtype)).sum());
      if(x.value_on[b].item<bool>())terms.push_back((v*x.value_root[b].narrow(1,0,old+count).to(dtype)).sum());
      if(x.bias_on[b].item<bool>())terms.push_back((log*x.bias_root[b].narrow(0,0,old+count).to(dtype)).sum());
      if(!terms.empty())expected=torch::autograd::grad({at::stack(terms).sum()},source,{},false,false,true);
    }
    same(input[b].narrow(0,0,count),icon[b].item<bool>(),expected[0],"source");
    for(I j=0;j<6;++j)same(observed[j][b],pc[b][j].item<bool>(),expected[j+1],"parameter "+std::to_string(j));
    same(key[b].narrow(1,0,old),cc[b][0].item<bool>(),expected[7],"old key");
    same(value[b].narrow(1,0,old),cc[b][1].item<bool>(),expected[8],"old value");
    same(bias[b].narrow(0,0,old),cc[b][2].item<bool>(),expected[9],"old log bias");
    for(auto a:{key[b].narrow(1,old,capacity-old),value[b].narrow(1,old,capacity-old),bias[b].narrow(0,old,capacity-old),
        input[b].narrow(0,count,input.size(1)-count)})
      if(!at::isfinite(a).all().item<bool>()||a.count_nonzero().item<I>())throw std::runtime_error("fiber adjoint padding leaked");
  }
}
void check(at::Device device,I width,I heads,int pool,int mode) {
  at::NoGradGuard guard;auto f=fixture(width,heads,pool,mode,0);auto device_f=device_input(f,device);
  auto error=at::zeros({1},device_f.rows.options().dtype(at::kInt));CannProgram p(device);p.limit_workspace(128*1024*1024);
  auto out=append_fiber_vjp(p,device_f,error,3,3,256*1024*1024);p.finish();
  for(int replay=0;replay<2;++replay) {
    if(replay) {
      f=fixture(width,heads,pool,mode,replay);auto next=device_input(f,device);auto dst=tensors(device_f),src=tensors(next);
      for(size_t i=0;i<dst.size();++i)dst[i]->copy_(*src[i]);
    }
    portable_torch::synchronize(device);p.run();if(error.cpu().item<int>())throw std::runtime_error("fiber VJP refused valid fixture");
    compare(f,out,at::kFloat);compare(f,out,at::kDouble);
    if(mode==0&&out.chunks.cpu().item<I>()!=0)throw std::runtime_error("all-None fiber replay executed source work");
  }
  device_f.ticks[0].fill_(device_f.max_repeat_ticks+1);portable_torch::synchronize(device);p.run();
  if(error.cpu().item<int>()!=2)throw std::runtime_error("unbounded fiber tick replay accepted");
  p.close();bool refused=false;
  try{CannProgram bad(device);append_fiber_vjp(bad,device_f,error,3,3,1);}catch(const std::invalid_argument&){refused=true;}
  if(!refused)throw std::runtime_error("fiber reverse memory budget ignored");
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("fiber VJP requires explicit NPU FP32");
    const auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(int pool=0;pool<5;++pool)for(int mode=0;mode<7;++mode) {
      try{check(device,4,2,pool,mode);++cases;}
      catch(...){std::cerr<<"fiber pool="<<pool<<" root="<<mode<<'\n';throw;}
    }
    for(I width:{1,257}){check(device,width,1,4,5);++cases;}
    std::cout<<"device-fiber-vjp: passed configurations="<<cases<<" replays="<<cases*2
      <<" CPU=FP32_FP64 complete_fiber=true None_zero=true independent_NPU_forward=true\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
