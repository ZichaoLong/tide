#include "full_vjp_fixture.h"
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/custom_function.h>
#include <iostream>
#include <limits>

namespace tide::device_online::test {
namespace {
at::Tensor round_half(const at::Tensor& x){return x+(x.detach().to(at::kHalf).to(at::kFloat)-x.detach());}
class HalfTanh : public torch::autograd::Function<HalfTanh> {
 public:
  static at::Tensor forward(torch::autograd::AutogradContext* ctx,const at::Tensor& x) {
    auto y=x.to(at::kHalf).tanh().to(at::kFloat);ctx->save_for_backward({y});return y;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list g) {
    const auto y=ctx->get_saved_variables()[0];return {g[0]*(1-y*y)};
  }
};
}
FullFixture full_fixture(int64_t width,int mode,bool tanh) {
  at::NoGradGuard guard;
  constexpr int64_t capacity=23,count=19,nodes=4,samples=2;
  const auto poison=std::numeric_limits<float>::quiet_NaN();
  auto meta=at::zeros({capacity,13},at::kLong),values=at::full({capacity,5*width+2},poison,at::kFloat);
  auto kinds=at::tensor({0,int(tanh),0,int(tanh)},at::kLong);
  auto gradient=at::full({capacity,width},poison,at::kFloat),connected=at::zeros({capacity},at::kBool);
  at::Tensor weights,biases;
  if(tanh) {
    weights=at::zeros({nodes+1,width,width},at::kFloat);biases=at::zeros({nodes+1,width},at::kFloat);
    for(int64_t n=0;n<nodes;++n) {
      if(!kinds[n].item<int64_t>()){weights[n].fill_(poison);biases[n].fill_(poison);continue;}
      weights[n].copy_(at::arange(width*width,at::kFloat).remainder(11).reshape({width,width})*.0005-.002+at::eye(width)*.125);
      biases[n].copy_(at::arange(width,at::kFloat).remainder(7)*.005-.013*n);
    }
  }
  for(int64_t row=0;row<count;++row) {
    const int64_t node=row%nodes,sample=(row/nodes)%samples,time=(int64_t(1)<<55)+17+row/nodes;
    const bool active=row%5!=0,on=active&&(mode==1||mode==3||mode==2&&row%3!=0);
    meta[row].narrow(0,0,4).copy_(at::tensor({sample,node,time,int64_t(active)},at::kLong));
    if(active) {
      // Recovering tanh as full-content loses its derivative information here.
      values[row].narrow(0,0,width).fill_(16777216.f);
      if(kinds[node].item<int64_t>())values[row].narrow(0,3*width,width).copy_(at::arange(width,at::kFloat).remainder(13)*.003-.011*row);
    }
    if(on){connected[row].fill_(true);gradient[row].copy_(mode==3?at::zeros({width},at::kFloat):
      at::arange(width,at::kFloat).remainder(5)*.03125-.0625+.015625*(row%3));}
  }
  // Poisoned padding is never part of this logical tape, even if its stale
  // connection bit is set. Empty/shorter replay must clear prior adjoints.
  connected[capacity-1].fill_(true);
  return {{meta,values,at::full({1},count,at::kLong),kinds,weights,biases,samples,width,tanh},gradient,connected};
}
std::vector<at::Tensor> full_reference(const FullFixture& f,at::ScalarType dtype) {
  at::AutoGradMode grad(true);const auto& t=f.tape;
  const auto capacity=t.values.size(0),nodes=t.kinds.size(0),count=t.count.item<int64_t>(),width=t.width;
  std::vector<at::Tensor> h,c,w,b,leaves,loss;
  auto leaf=[&](const at::Tensor& x){return x.detach().to(dtype).to(dtype==at::kHalf?at::kFloat:dtype).clone().set_requires_grad(true);};
  for(int64_t i=0;i<capacity;++i){h.push_back(leaf(t.values[i].narrow(0,0,width)));c.push_back(leaf(t.values[i].narrow(0,3*width,width)));}
  for(int64_t n=0;n<nodes;++n) {
    w.push_back(leaf(t.has_tanh?t.weights[n]:at::zeros({width,width},at::kFloat)));
    b.push_back(leaf(t.has_tanh?t.biases[n]:at::zeros({width},at::kFloat)));
  }
  for(int64_t i=0;i<count;++i)if(f.connected[i].item<bool>()) {
    const auto n=t.metadata[i][1].item<int64_t>();auto y=h[i];
    if(t.kinds[n].item<int64_t>()) {
      auto product=at::matmul(c[i],w[n]);
      if(dtype==at::kHalf) {
        const auto rounded=at::matmul(c[i].detach().to(at::kHalf),w[n].detach().to(at::kHalf)).to(at::kFloat);
        product=product+(rounded-product.detach());
        y=round_half(y+HalfTanh::apply(round_half(product+b[n])));
      } else y=y+at::tanh(product+b[n]);
    }
    loss.push_back((y*f.gradient[i].to(dtype==at::kHalf?at::kFloat:dtype)).sum());
  }
  for(const auto& group:{h,c,w,b})leaves.insert(leaves.end(),group.begin(),group.end());
  if(loss.empty())return std::vector<at::Tensor>(leaves.size());
  return torch::autograd::grad({at::stack(loss).sum()},leaves,{},false,false,true);
}
void full_same_precision(const at::Tensor& value,const at::Tensor& connected,const at::Tensor& expected,const char* field,bool half) {
  if(connected.item<bool>()!=expected.defined())throw std::runtime_error(std::string(field)+" connection mismatch");
  if(expected.defined()) {
    if(!at::allclose(value,expected.to(at::kFloat),half?2e-3:1e-5,half?2e-5:1e-6)) {
      std::cerr<<field<<" max_abs_error="<<(value-expected.to(at::kFloat)).abs().max().item<double>()<<'\n';
      if(value.numel()<=8)std::cerr<<"actual="<<value<<" expected="<<expected.to(at::kFloat)<<'\n';
      throw std::runtime_error(std::string(field)+" numerical mismatch");
    }
  } else if(!at::equal(value,at::zeros_like(value)))throw std::runtime_error(std::string(field)+" evaluated absent poison");
}
void full_same(const at::Tensor& value,const at::Tensor& connected,const at::Tensor& expected,const char* field) {
  full_same_precision(value,connected,expected,field,false);
}
void full_compare(const FullFixture& f,const FullVjp& out,at::ScalarType dtype) {
  auto expected=full_reference(f,dtype);
  const auto capacity=f.tape.values.size(0),nodes=f.tape.kinds.size(0);
  auto h=out.content.cpu(),c=out.comparison.cpu(),hc=out.content_connected.cpu(),cc=out.comparison_connected.cpu();
  auto wc=out.parameter_connected.cpu();
  for(int64_t i=0;i<capacity;++i){full_same_precision(h[i],hc[i],expected[i],"Full content",dtype==at::kHalf);full_same_precision(c[i],cc[i],expected[capacity+i],"Full comparison",dtype==at::kHalf);}
  if(f.tape.has_tanh) {
    auto w=out.weights.cpu(),b=out.biases.cpu();
    for(int64_t n=0;n<nodes;++n){full_same_precision(w[n],wc[n],expected[2*capacity+n],"Full weight",dtype==at::kHalf);full_same_precision(b[n],wc[n],expected[2*capacity+nodes+n],"Full bias",dtype==at::kHalf);}
  } else if(out.weights.defined()||out.biases.defined()||wc.any().item<bool>())throw std::runtime_error("identity Full fabricated parameters");
}
} // namespace tide::device_online::test
