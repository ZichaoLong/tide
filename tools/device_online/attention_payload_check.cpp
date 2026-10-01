#include "tiled_attention.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace tide::device_online;
using I=int64_t;
void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
constexpr I capacity=300,owners=2,events_count=3,queries=4;
const float poison=std::numeric_limits<float>::quiet_NaN();
struct Case {
  TiledAttentionSpec spec;
  at::Tensor events,tokens,ids,q,key,value,bias;
  std::vector<std::vector<I>> visible;
  std::vector<I> selected{2,-1,0,1};
  I heads,kv_heads,width,d;
  Case(I w,I h,I kh,I mode,I tile,at::ScalarType dtype):heads(h),kv_heads(kh),width(w),d(w/h) {
    const bool compact=mode==1,fiber=mode>=2;
    spec={h,kh,capacity,owners,tile,fiber,1./std::sqrt(double(d)),compact?events_count:0,mode==3};
    events=at::zeros({events_count,7},at::kLong);tokens=at::zeros({events_count,4},at::kLong);
    visible.resize(events_count);
    if(compact) {
      // Two successive events for owner0, including a sliding window, then
      // a different owner. Public inputs define the visible CPU oracle sets.
      events=at::tensor({0,0,0,5,6,5,0, 1,0,0,6,4,6,0,
                        2,1,0,257,258,257,2},at::kLong).reshape({3,7});
      visible[0]={0,1,2,3,4,600};visible[1]={3,4,600,601};
      for(I row=300;row<557;++row)visible[2].push_back(row);visible[2].push_back(602);
    }else {
      events=at::tensor({0,0,0,0,1,0,0, 1,0,0,0,5,0,0,
                        2,1,0,0,257,0,0},at::kLong).reshape({3,7});
      visible[0]={0};visible[1]={0,1,2,3,4};
      for(I row=300;row<557;++row)visible[2].push_back(row);
    }
    for(I i=0;i<events_count;++i)tokens[i][1].fill_(i);
    ids=at::tensor(selected,at::kLong);
    const I rows=owners*capacity+spec.event_rows+1;
    q=((at::arange(queries*w,at::kFloat).remainder(13)-6)*.125f).reshape({queries,w}).to(dtype);
    q[1].fill_(poison); // Absent query must never enter normalization.
    key=((at::arange(rows*kh*d,at::kFloat).remainder(17)-8)*.0625f).reshape({rows,kh,d}).to(dtype);
    value=((at::arange(rows*kh*d,at::kFloat).remainder(23)-11)*.125f).reshape({rows,kh,d}).to(dtype);
    key[-1].zero_();value[-1].zero_();
    bias=spec.fiber_bias_rows?at::zeros({events_count,capacity},at::kFloat):at::zeros({rows},at::kFloat);
    if(fiber)bias.copy_((at::arange(bias.numel(),at::kFloat).remainder(7)*-.0625f).reshape(bias.sizes()));
  }
  at::Tensor expected(bool fp64) const {
    auto result=at::zeros_like(q);
    for(I row=0;row<queries;++row) {
      const I e=selected[row];if(e<0)continue;
      auto index=at::tensor(visible[e],at::kLong);
      const auto kk=key.index_select(0,index),vv=value.index_select(0,index);
      for(I head=0;head<heads;++head) {
        const auto dot_type=fp64?at::kDouble:q.scalar_type();
        auto query=q[row].narrow(0,head*d,d).to(dot_type);
        auto keys=kk.select(1,head/(heads/kv_heads)).to(dot_type);
        // A dense CPU expression normalizes the full visible set at once; it
        // does not replay device tiles or reuse candidate-produced metadata.
        auto logits=at::matmul(keys,query).to(fp64?at::kDouble:at::kFloat)*spec.scale;
        if(spec.fiber)logits+=spec.fiber_bias_rows?bias[e].narrow(0,0,index.numel()):bias.index_select(0,index);
        auto values=vv.select(1,head/(heads/kv_heads)).to(logits.scalar_type());
        result[row].narrow(0,head*d,d).copy_(at::matmul(at::softmax(logits,0),values));
      }
    }
    return result;
  }
};
void check(at::Device device,at::ScalarType dtype) {
  I cases=0,replays=0,empty=0,refusals=0;
  for(const auto& geometry:std::vector<std::vector<I>>{{1,1,1},{4,4,1},{33,3,1},{257,1,1}})
  for(I mode:{0,1,2,3})for(I tile:{1,7,128,256}) {
    Case c(geometry[0],geometry[1],geometry[2],mode,tile,dtype);
    auto events=c.events.to(device),tokens=c.tokens.to(device),ids=c.ids.to(device),q=c.q.to(device);
    auto key=c.key.to(device),value=c.value.to(device),bias=c.bias.to(device);
    auto error=at::zeros({1},ids.options().dtype(at::kInt)),work=at::zeros({3},ids.options());
    CannProgram p(device);auto result=append_tiled_attention(p,events,tokens,ids,q,key,value,bias,error,work,c.spec);p.finish();
    for(I replay=0;replay<3;++replay) {
      if(replay==1){c.q.mul_(-1);c.value.mul_(-.5);c.bias.mul_(2);}
      if(replay==2){c.q.zero_();c.bias.zero_();c.value.fill_(40000);c.value[-1].zero_();}
      q.copy_(c.q);value.copy_(c.value);bias.copy_(c.bias);work.zero_();
      portable_torch::synchronize(device);p.run();auto actual=result.cpu();
      require(actual.scalar_type()==dtype,"attention changed payload dtype");
      require(error.cpu().item<int>()==0,"attention unexpectedly refused");
      for(bool fp64:{false,true})require(at::allclose(actual,c.expected(fp64),dtype==at::kHalf?3e-3:2e-5,dtype==at::kHalf?1e-3:2e-6),
        "tiled attention differs from independent dense CPU expression");
      require(actual[1].eq(0).all().item<bool>(),"absent query was evaluated");
      if(replay==2)for(I row:{0,2,3})require(at::allclose(actual[row].to(at::kFloat),at::full({c.width},40000.f),0.,dtype==at::kHalf?0.:.05),"unnormalized tile overflow/denominator loss");
      const I maximum=c.visible[2].size();I entries=0;for(const auto& visible:c.visible)entries+=visible.size()*c.heads;
      auto counters=work.cpu();const I tiles=(maximum+tile-1)/tile;
      require(counters[0].item<I>()==tiles&&counters[1].item<I>()==entries
        &&counters[2].item<I>()==tiles*queries*c.heads*tile-entries,"actual/padding key work differs");
      ++replays;
    }
    // A replay with no queries reuses the same captured program and buffers.
    ids.fill_(-1);work.zero_();portable_torch::synchronize(device);p.run();
    require(result.cpu().eq(0).all().item<bool>()&&work.cpu().eq(0).all().item<bool>(),"empty replay retained stale values");++empty;
    error.fill_(9);ids.copy_(c.ids);portable_torch::synchronize(device);p.run();
    require(error.cpu().item<int>()==9&&result.cpu().eq(0).all().item<bool>(),"sticky failure executed attention");++refusals;
    ++cases;
  }
  std::cout<<"attention-payload: passed configurations="<<cases<<" replays="<<replays
    <<" empty="<<empty<<" sticky_refusals="<<refusals
    <<" oracle=dense_CPU_payload_and_FP64 score_accumulator=FP32 scope=component\n";
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))throw std::invalid_argument("attention payload requires explicit NPU FP32/FP16");
    args.allow_npu_float16=true;auto device=portable_torch::resolve_device(args);
    if(device.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("attention payload requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    check(device,args.dtype);runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
