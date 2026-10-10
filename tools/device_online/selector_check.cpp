#include "device_backend.h"
#include "frame_selector.h"
#include "portable_torch/runtime.hpp"
#include "tide/region.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>

namespace {
using namespace tide::device_online;
using I=int64_t;
void require(bool yes,const char* text){if(!yes)throw std::runtime_error(text);}
SelectionHistory cpu_copy(const SelectionHistory& h){return {h.counts.cpu().clone(),h.seen.cpu().clone(),h.last_time.cpu().clone(),h.present.cpu().clone()};}
void same(const SelectionHistory& a,const SelectionHistory& b) {
  require(at::equal(a.counts.cpu(),b.counts.cpu())&&at::equal(a.seen.cpu(),b.seen.cpu())
      &&at::equal(a.last_time.cpu(),b.last_time.cpu())&&at::equal(a.present.cpu(),b.present.cpu()),"selection history differs");
}
void check(at::Device device) {
  at::NoGradGuard guard;constexpr I capacity=16,samples=2,nodes=5,regions=2;
  const std::vector<I> owners{0,1,0,1,0};
  const auto opts=at::TensorOptions().device(device).dtype(at::kLong);
  I cases=0;
  for(int variant=0;variant<4;++variant) {
    std::vector<SelectionPolicy> policies{{variant==3?0:1+variant,variant!=1,variant>=2},{variant==0?0:2,variant%2==0,variant%2!=0}};
    FrameSelector selector(owners,policies,samples,device);auto history=selector.initial();
    auto expected=cpu_copy(history);
    expected.counts[0][2].fill_((I(1)<<55)+3);expected.seen[0][2].fill_(true);
    expected.counts[1][4].fill_((I(1)<<55)+1);expected.seen[1][4].fill_(true);
    expected.seen[1][0].fill_(true); // Explicit present zero differs from missing.
    history.counts.copy_(expected.counts);history.seen.copy_(expected.seen);
    ReadyBatch ready;ready.fibers=at::zeros({capacity,4},opts);ready.frames=at::zeros({capacity,3},opts);
    ready.frame_offsets=at::zeros({capacity+1},opts);ready.frame_fibers=at::zeros({capacity},opts);ready.counts=at::zeros({3},opts);
    auto scores=at::zeros({capacity},opts.dtype(at::kFloat)),error=at::zeros({1},opts.dtype(at::kInt)),refusal=at::zeros_like(error);
    DeviceProgram program(device);auto out=selector.append_stage(program,ready,scores,history,error);
    program.add(error,refusal);selector.append_commit(program,history,out,error);program.finish();
    auto fibers=at::zeros({capacity,4},at::kLong),frames=at::zeros({capacity,3},at::kLong);
    auto offsets=at::zeros({capacity+1},at::kLong),members=at::zeros({capacity},at::kLong);
    I nf=0,ng=0;
    for(I round=0;round<5;++round) {
      fibers.zero_();frames.zero_();offsets.zero_();members.zero_();nf=0;ng=0;
      auto desc=at::zeros({capacity},at::kFloat),active=at::zeros({capacity},at::kBool),controls=at::zeros_like(desc);
      for(I b=0;b<samples;++b)for(I r=0;r<regions;++r)for(I step=0;step<(r==0?2:1);++step) {
        I time=(I(1)<<55)+round*8+step;
        frames[ng].copy_(at::tensor({b,r,time},at::kLong));offsets[ng].fill_(nf);
        std::vector<tide::Candidate> candidates;std::vector<I> membership,indices;
        for(I n=0;n<nodes;++n)if(owners[n]==r)membership.push_back(n);
        for(auto n:membership) {
          if(step==1&&n==4)continue;
          fibers[nf].copy_(at::tensor({b,n,time,ng},at::kLong));members[nf].fill_(nf);
          float score=round%3==0?1:round%3==1?float(n-2)*0.5f:-float(n+1);
          desc[nf].fill_(score);candidates.push_back({n,desc[nf]});indices.push_back(nf);++nf;
        }
        tide::Region spec;spec.budget=policies[r].budget;spec.count_priority=policies[r].count_priority;
        spec.selector=policies[r].positive_only?"positive-v1":"count-v1";
        tide::History prior;prior.node_maps["selected"]={};prior.last_time=expected.last_time[b][r].item<I>();
        for(auto n:membership)if(expected.seen[b][n].item<bool>())prior.node_maps["selected"][n]=expected.counts[b][n].item<I>();
        auto reference=tide::make_region_kernel(spec)->step({}, {prior,time,candidates,{spec,membership},desc.options()});
        for(size_t j=0;j<candidates.size();++j) {
          I n=candidates[j].node,f=indices[j];active[f].fill_(reference.active.count(n)!=0);
          controls[f].copy_(reference.controls.at(n));
        }
        for(auto [n,count]:reference.history.node_maps.at("selected")){expected.counts[b][n].fill_(count);expected.seen[b][n].fill_(true);}
        expected.last_time[b][r].fill_(reference.history.last_time);expected.present[b][r].fill_(true);++ng;
      }
      offsets.slice(0,ng).fill_(nf);
      ready.fibers.copy_(fibers);ready.frames.copy_(frames);ready.frame_offsets.copy_(offsets);ready.frame_fibers.copy_(members);
      ready.counts.copy_(at::tensor({I(0),nf,ng},at::kLong));scores.copy_(desc);
      portable_torch::synchronize(device);program.run();
      require(error.cpu().item<int>()==0,"selector refused legal frame sequence");
      require(at::equal(out.active.cpu(),active),"count/score/node selection or stable ties differ");
      require(at::allclose(out.controls.cpu(),controls,1e-5,1e-6),"complete-candidate softmax differs");
      same(history,expected);++cases;
    }
    // No candidates leaves even explicit-zero/missing history structure intact.
    ready.counts.zero_();ready.frame_offsets.zero_();portable_torch::synchronize(device);program.run();
    same(history,expected);require(!out.active.cpu().any().item<bool>()&&out.branch.cpu().item<int>()==0,"empty selection mutated activity");++cases;
    // One single frame for refusal/error tests, at a strictly newer exact time.
    ready.counts.copy_(at::tensor({0,1,1},at::kLong));ready.fibers.zero_();ready.frames.zero_();ready.frame_fibers.zero_();
    ready.fibers[0][2].fill_((I(1)<<55)+99);ready.frames[0][2].fill_((I(1)<<55)+99);ready.frame_offsets.slice(0,1).fill_(1);scores.fill_(1);
    refusal.fill_(1);portable_torch::synchronize(device);program.run();
    require(error.cpu().item<int>()==1,"downstream refusal was lost");same(history,expected);++cases;
    refusal.zero_();error.zero_();scores[0].fill_(std::numeric_limits<float>::quiet_NaN());
    portable_torch::synchronize(device);program.run();require(error.cpu().item<int>()==6,"nonfinite score accepted");same(history,expected);++cases;
    error.zero_();scores.fill_(1);ready.frame_fibers[0].fill_(-1);
    portable_torch::synchronize(device);program.run();require(error.cpu().item<int>()==2,"invalid frame index accepted");same(history,expected);++cases;
    if(policies[0].budget>0) {
      error.zero_();ready.frame_fibers.zero_();history.counts[0][0].fill_(std::numeric_limits<I>::max());history.seen[0][0].fill_(true);
      expected=cpu_copy(history);portable_torch::synchronize(device);program.run();
      require(error.cpu().item<int>()==5,"selected-count overflow accepted");same(history,expected);++cases;
    }
    program.close();
  }
  std::cout<<"device-selector: passed cases="<<cases<<" exact_counts=true stable_ties=true complete_denominators=true"
    <<" history_continuation=true atomic_refusal=true scope=count_and_positive_region_contracts_only\n";
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("selector check requires explicit NPU and FP32 scoring");
    auto device=portable_torch::resolve_device(args);
    if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("selector check requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);check(device);runtime.close();return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
