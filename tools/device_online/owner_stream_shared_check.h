#pragma once
// Included by owner_stream_check.cpp after its comparison helpers.
namespace {
void shared_sequence(const std::vector<at::Device>& devices) {
  const int64_t size=262147,capacity=262139;
  const std::vector<float> values{16777216.f,1.f,-16777216.f};
  std::vector<std::vector<Tensor>> inputs(2),flags(2);
  std::vector<Tensor> outputs,connected,errors,send,receive;
  for(auto d:devices) {
    auto f=at::TensorOptions().device(d).dtype(at::kFloat);
    outputs.push_back(at::zeros({size},f));connected.push_back(at::zeros({1},f.dtype(at::kBool)));
    errors.push_back(at::zeros({1},f.dtype(at::kInt)));
  }
  for(size_t d=0;d<2;++d)for(auto value:values) {
    inputs[d].push_back(at::full({size},value,outputs[d].options()));
    flags[d].push_back(at::ones({1},connected[d].options()));
  }
  std::vector<int64_t> baseline;
  for(auto d:devices) {
    portable_torch::synchronize(d);
    baseline.push_back(c10_npu::NPUCachingAllocator::getDeviceStats(d.index()).allocated_bytes[0].current);
    c10_npu::NPUCachingAllocator::resetPeakStats(d.index());
  }
  std::vector<std::unique_ptr<CannProgram>> programs;
  for(size_t d=0;d<2;++d) {
    send.push_back(at::zeros({capacity},outputs[d].options()));receive.push_back(at::zeros_like(send.back()));
    programs.push_back(std::make_unique<CannProgram>(devices[d]));programs.back()->limit_workspace(1024*1024);
  }
  const std::vector<std::pair<size_t,size_t>> pairs{{0,1},{1,0},{0,0},{1,1}};
  std::vector<OwnerStream> streams;int64_t metadata=0;
  for(size_t ordinal=0;ordinal<9;++ordinal)for(auto [from,to]:pairs) {
    const bool peer=from!=to;
    OwnerStreamField field{{inputs[from][ordinal%3],flags[from][ordinal%3]},{{outputs[to],at::kFloat}},connected[to]};
    const auto overhead=owner_stream_metadata_bytes(1,1,peer),budget=overhead+(peer?8:4)*capacity;
    OwnerStreamPackets shared{send[from],peer?receive[to]:Tensor{}};
    if(ordinal==0) {
      auto invalid=shared;invalid.send=shared.send.narrow(0,0,capacity-1);
      reject([&]{append_owner_stream(*programs[from],*programs[to],{field},errors[from],errors[to],budget,true,invalid);});
      invalid=shared;invalid.receive=peer?Tensor{}:receive[to];
      reject([&]{append_owner_stream(*programs[from],*programs[to],{field},errors[from],errors[to],budget,true,invalid);});
    }
    auto plan=append_owner_stream(*programs[from],*programs[to],{field},errors[from],errors[to],budget,true,shared);
    require(plan.reserved_bytes==overhead&&plan.capacity==capacity&&plan.iterations==2,"shared packet accounting differs");
    metadata+=overhead;streams.push_back(std::move(plan));
  }
  for(auto& p:programs)p->finish();
  for(int replay=0;replay<5;++replay) {
    std::vector<Tensor> expected{at::zeros({size}),at::zeros({size})};
    for(size_t d=0;d<2;++d) {
      outputs[d].zero_();connected[d].zero_();errors[d].fill_(replay==3?29:0);
      for(size_t k=0;k<3;++k) {
        const bool live=replay!=2||k!=1;
        inputs[d][k].fill_(live?(replay==4?0.f:values[k]):std::numeric_limits<float>::quiet_NaN());
        flags[d][k].fill_(live);
      }
    }
    if(replay!=3)for(size_t ordinal=0;ordinal<9;++ordinal)for(auto [from,to]:pairs)
      if(replay!=2||ordinal%3!=1)expected[to].add_(replay==4?0.f:values[ordinal%3]);
    for(auto d:devices)portable_torch::synchronize(d);
    for(auto& p:programs)p->submit();for(auto& p:programs)p->wait();
    for(size_t d=0;d<2;++d) {
      require(at::equal(outputs[d].cpu(),expected[d]),"shared cross-pair order/tail differs");
      require(connected[d].cpu().item<bool>()==(replay!=3),"shared None/zero connection differs");
      require(errors[d].cpu().item<int>()==(replay==3?29:0),"shared replay error differs");
    }
  }
  int64_t peak=0;for(size_t d=0;d<2;++d)
    peak+=c10_npu::NPUCachingAllocator::getDeviceStats(devices[d].index()).allocated_bytes[0].peak-baseline[d];
  const int64_t reserved=4*4*capacity+metadata;
  require(peak<reserved+12*1024*1024,"shared packets allocated per group instead of per device");
  std::cout<<"owner-stream-shared {\"groups\":36,\"replays\":5,\"reserved_bytes\":"<<reserved
    <<",\"peak_allocated_delta\":"<<peak<<"}\n";
  for(auto& p:programs)p->close();for(auto& s:streams)if(s.peer)s.peer->close();
}
}
