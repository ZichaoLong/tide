#include "aggregate_vjp.h"
#include "full_vjp_fixture.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <array>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

namespace {
using namespace tide;using namespace tide::device_online;
const std::array<std::string,5> names{"sum","mean","weighted_mean","active_softmax","all_softmax"};
struct Fixture {
  Graph graph;ReverseTape tape;ReverseLinks links;Tensor gradient,connected,count,range;
  std::vector<std::vector<Index>> fibers;
  std::vector<Index> source;
};
Fixture fixture(Index d,Index slots,int kind,int mode,at::ScalarType payload) {
  const float nan=std::numeric_limits<float>::quiet_NaN();Fixture f;
  f.graph.nodes={{0},{0},{0}};f.graph.regions={{3}};
  for(Index n=0;n<3;++n){f.graph.nodes[n].aggregation=names[kind];for(Index s=0;s<slots;++s)f.graph.inputs.push_back(n);}
  f.graph.compile();auto& t=f.tape;
  constexpr Index capacity=9,rows=6,fibers=24;
  t.graph=&f.graph;t.state.metadata=at::zeros({capacity,13},at::kLong);t.full.width=d;
  t.fiber_values=at::full({fibers,d},nan,at::kFloat);
  t.sources=at::empty({3*slots,2},at::kLong);t.source_scales=at::ones({3*slots},at::kFloat);
  t.aggregate={at::full({3},kind,at::kLong),at::full({3},slots,at::kLong),at::zeros({3,slots},at::kFloat),slots};
  for(Index n=0;n<3;++n)for(Index s=0;s<slots;++s) {
    // Physical source order is a permutation of the logical coefficient domain.
    t.sources[n*slots+s].copy_(at::tensor({n,(s+1)%slots},at::kLong));
    // Logical slot0 has nonzero raw data behind a zero physical scale;
    // logical slot1 separately carries a present-zero message.
    t.source_scales[n*slots+s].fill_(s==slots-1?0.f:.5f+.125f*(s%3));
    t.aggregate.weights[n][s].fill_(n==2?nan:.125f*(s%7)-.25f);
  }
  f.links.messages=at::full({fibers,4},-1,at::kLong);
  f.links.consumer_head=at::full({capacity},-1,at::kLong);f.links.consumer_next=at::full({fibers},-1,at::kLong);
  f.links.fibers=fibers;f.links.pending=0;f.links.outputs=0;f.links.parameters=3*slots;
  f.count=at::full({1},rows,at::kLong);f.range=at::tensor(std::vector<Index>{1,1+rows},at::kLong);
  f.gradient=at::full({capacity,d},nan,at::kFloat);f.connected=at::zeros({capacity},at::kBool);
  f.fibers.resize(rows);f.source.assign(fibers,-1);Index message=0;
  for(Index row=0;row<rows;++row) {
    const Index node=row%3;t.state.metadata[row+1][1].fill_(node);
    const bool on=node!=2&&mode!=0;f.connected[row].fill_(on);
    if(on)f.gradient[row].copy_(mode==2?at::zeros({d},at::kFloat):at::arange(d,at::kFloat).remainder(7)*.015625f-.03125f);
    std::vector<Index> present{0,slots-1};if(row%2)present.push_back(1);
    for(auto slot:present) {
      const auto physical=node*slots+(slot+slots-1)%slots;
      const auto value=slot==1?at::zeros({d},at::kFloat):at::arange(d,at::kFloat).remainder(5)*.03125f+.125f*(row+1);
      t.fiber_values[message].copy_(node==2?at::full({d},nan,at::kFloat):value);
      f.links.messages[message][2].fill_(physical);f.source[message]=physical;f.fibers[row].push_back(message++);
    }
    for(auto it=f.fibers[row].rbegin();it!=f.fibers[row].rend();++it) {
      f.links.consumer_next[*it].copy_(f.links.consumer_head[row+1]);f.links.consumer_head[row+1].fill_(*it);
    }
  }
  if(kind==2||kind==3)for(Index n=0;n<2;++n)for(Index slot=2;slot<slots-1;++slot)t.aggregate.weights[n][slot].fill_(nan);
  if(payload==at::kHalf) {
    // Non-dyadic values exercise the source-product rounding boundary; retain
    // present-zero messages and zero scales as separate connected cases.
    t.fiber_values=at::where(t.fiber_values.ne(0),t.fiber_values+.00091f,t.fiber_values).to(at::kHalf).to(at::kFloat);
    t.source_scales=at::where(t.source_scales.ne(0),t.source_scales+.03113f,t.source_scales).to(at::kHalf);
    t.aggregate.weights=t.aggregate.weights.to(at::kHalf).to(at::kFloat);
    f.gradient=f.gradient*256.f;
  }
  f.connected[capacity-1].fill_(true); // Stale tail must never become a row.
  return f;
}
std::vector<Tensor> reference(const Fixture& f,at::ScalarType dtype,bool quantized=true) {
  at::AutoGradMode enabled(true);const auto& t=f.tape;const auto slots=t.aggregate.slots;
  std::vector<Tensor> x,scales,weights,leaves,terms;
  auto leaf=[&](const Tensor& v){return v.detach().to(dtype).clone().set_requires_grad(true);};
  for(Index i=0;i<t.fiber_values.size(0);++i)x.push_back(leaf(t.fiber_values[i]));
  for(Index i=0;i<t.source_scales.numel();++i)scales.push_back(leaf(t.source_scales[i]));
  for(Index i=0;i<3*slots;++i)weights.push_back(leaf(t.aggregate.weights.reshape({-1})[i]));
  for(Index row=0;row<f.count.item<Index>();++row)if(f.connected[row].item<bool>()) {
    const Index node=t.state.metadata[row+1][1].item<Index>(),kind=t.aggregate.kinds[node].item<Index>();
    std::vector<Index> present;for(auto m:f.fibers[row])present.push_back(t.sources[f.source[m]][1].item<Index>());
    auto domain=present;if(kind==4){domain.clear();for(Index i=0;i<slots;++i)domain.push_back(i);}
    std::map<Index,Tensor> coefficient;
    if(kind==1)for(auto slot:present)coefficient[slot]=at::full({},1./present.size(),at::TensorOptions().dtype(dtype));
    else {
      std::vector<Tensor> raw;for(auto slot:domain)raw.push_back(weights[node*slots+slot]);
      auto z=at::stack(raw);auto p=kind==2?at::softplus(z):at::softmax(z,0);if(kind==2)p=p/p.sum();
      for(size_t i=0;i<domain.size();++i)coefficient[domain[i]]=p[i];
    }
    std::vector<Tensor> contributions;for(size_t i=0;i<f.fibers[row].size();++i) {
      const auto m=f.fibers[row][i];auto weighted=x[m]*scales[f.source[m]];
      if(quantized&&t.source_scales.scalar_type()==at::kHalf)
        weighted=weighted+(weighted.detach().to(at::kHalf).to(dtype)-weighted.detach());
      contributions.push_back(weighted*coefficient[present[i]]);
    }
    terms.push_back((at::stack(contributions).sum(0)*f.gradient[row].to(dtype)).sum());
  }
  for(const auto& group:{x,scales,weights})leaves.insert(leaves.end(),group.begin(),group.end());
  if(terms.empty())return std::vector<Tensor>(leaves.size());
  return torch::autograd::grad({at::stack(terms).sum()},leaves,{},false,false,true);
}
void run(at::Device device,Index d,Index slots,int kind,int mode,at::ScalarType payload) {
  at::NoGradGuard guard;auto f=fixture(d,slots,kind,mode,payload);f.tape.graph=&f.graph;
  if(payload==at::kHalf&&d==257&&kind==2&&mode==1) {
    auto exact=reference(f,at::kFloat),wrong=reference(f,at::kFloat,false);bool different=false;
    for(size_t i=0;i<exact.size();++i)if(exact[i].defined()&&!at::allclose(exact[i],wrong[i],1e-5,1e-6))different=true;
    if(!different)throw std::runtime_error("Aggregate half fixture does not distinguish unrounded source products");
  }
  auto t=f.tape;auto links=f.links;
  for(auto* x:{&t.state.metadata,&t.fiber_values,&t.sources,&t.source_scales,&t.aggregate.kinds,&t.aggregate.lengths,&t.aggregate.weights,
      &links.messages,&links.consumer_head,&links.consumer_next})*x=x->to(device);
  auto count=f.count.to(device),range=f.range.to(device),gradient=f.gradient.to(device),connected=f.connected.to(device);
  auto messages=at::zeros_like(t.fiber_values),partials=at::zeros_like(messages),error=at::zeros({1},messages.options().dtype(at::kInt));
  AggregateVjp out{at::zeros_like(t.aggregate.weights),at::zeros({3,slots},connected.options()),at::zeros({1},count.options())};
  if(d==1&&kind==1&&mode==0) {
    bool refused=false;try {CannProgram small(device);
      append_aggregate_vjp(small,t,links,count,range,gradient,connected,messages,partials,out,error,1,1);
    }catch(const std::invalid_argument&){refused=true;}
    if(!refused)throw std::runtime_error("Aggregate VJP ignored tensor budget");
  }
  CannProgram p(device);p.limit_workspace(32*1024*1024);
  for(const auto& x:{messages,partials,out.values,out.connected,out.chunks})p.zero(x);
  append_aggregate_vjp(p,t,links,count,range,gradient,connected,messages,partials,out,error,mode==1?4:1,32*1024*1024);p.finish();
  for(Index rows:{6,2,0}) {
    f.count.fill_(rows);count.copy_(f.count);portable_torch::synchronize(device);p.run();
    if(error.cpu().item<int>())throw std::runtime_error("Aggregate VJP refused valid domain");
    auto dx=messages.cpu(),parts=partials.cpu(),dw=out.values.cpu(),on=out.connected.cpu();
    std::vector<bool> live(f.source.size(),false);for(Index r=0;r<rows;++r)if(f.connected[r].item<bool>())for(auto m:f.fibers[r])live[m]=true;
    for(auto dtype:{at::kFloat,at::kDouble}) {
      const auto expected=reference(f,dtype);Index index=0;
      for(Index i=0;i<dx.size(0);++i)test::full_same(dx[i],at::full({},bool(live[i]),at::kBool),expected[index++],"Aggregate message");
      for(Index i=0;i<t.source_scales.numel();++i) {
        Tensor value=at::zeros({},at::kFloat);bool used=false;
        for(size_t m=0;m<f.source.size();++m)if(f.source[m]==i&&live[m]){value=value+parts[m].sum();used=true;}
        test::full_same(value,at::full({},used,at::kBool),expected[index++],"Aggregate physical scale");
      }
      for(Index n=0;n<3;++n)for(Index slot=0;slot<slots;++slot)
        test::full_same(dw[n][slot],on[n][slot],expected[index++],"Aggregate coefficient");
    }
  }
  if(d==1&&mode==1) {
    // Two physical messages must never collapse into one logical slot, even
    // when their payload or upstream derivative happens to be zero.
    auto duplicate=f.tape.sources.clone();duplicate[slots-2][1].fill_(0);
    t.sources.copy_(duplicate);count.fill_(6);portable_torch::synchronize(device);p.run();
    if(error.cpu().item<int>()!=2)throw std::runtime_error("Aggregate VJP accepted duplicate logical source");
  }
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("Aggregate VJP requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);int cases=0;
    for(int kind=1;kind<5;++kind)for(Index d:{1,7,257})for(int mode:{0,1,2}) {
      try{run(device,d,5,kind,mode,args.dtype);++cases;}catch(...){std::cerr<<"Aggregate kind="<<kind<<" width="<<d<<" mode="<<mode<<'\n';throw;}
    }
    for(int kind:{2,3,4}){run(device,7,257,kind,1,args.dtype);++cases;}
    std::cout<<"device-aggregate-vjp: passed cases="<<cases<<" replays="<<cases*3<<" CPU=FP32_FP64 None_zero=true absent_domains=true\n";
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
