#include "owner_stream.h"
#include "sharded_parameter_reduce.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <iostream>
#include <limits>
namespace {
using namespace tide;using namespace tide::device_online;
void require(bool v,const char* why){if(!v)throw std::runtime_error(why);}
void reject(const std::function<void()>& f) {
  bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}
  require(caught,"invalid owner stream was accepted");
}
// A packet crosses field and strided-row boundaries. Payload/flags change on
// every replay; no host callback participates in the device packet loop.
void stream_case(at::Device from,at::Device to,int64_t capacity,bool accumulate) {
  const bool peer=from!=to;auto f=at::TensorOptions().device(from).dtype(at::kFloat);
  auto error=at::zeros({1},f.dtype(at::kInt)),remote=at::zeros({1},error.options().device(to));
  CannProgram send(from);std::unique_ptr<CannProgram> receive;
  if(peer)receive=std::make_unique<CannProgram>(to);auto& dest=peer?*receive:send;
  std::vector<OwnerStreamField> fields;std::vector<Tensor> storage,expected;
  const std::vector<int64_t> sizes{771,17,33};int64_t writes=0,total=0;
  for(size_t i=0;i<sizes.size();++i) {
    const int64_t rows=i==0?3:1,cols=sizes[i]/rows;
    OwnerStreamField field{{at::zeros({sizes[i]},f),at::zeros({1},f.dtype(at::kBool))},{},
      accumulate?at::zeros({1},f.device(to).dtype(at::kBool)):Tensor{}};
    const int aliases=accumulate?1:3;
    for(int a=0;a<aliases;++a) {
      auto bank=at::empty({rows,cols+5},f.device(to).dtype(a==1?at::kHalf:at::kFloat));
      storage.push_back(bank);field.destinations.push_back({bank.narrow(1,2,cols),a?at::kHalf:at::kFloat});++writes;
    }
    fields.push_back(std::move(field));total+=sizes[i];
  }
  const auto metadata=owner_stream_metadata_bytes(fields.size(),writes,peer),budget=metadata+(peer?8:4)*capacity;
  reject([&]{append_owner_stream(send,dest,fields,error,remote,metadata+(peer?8:4)-1,accumulate);});
  auto wrong=fields;wrong[0].source.values=Tensor{};
  reject([&]{append_owner_stream(send,dest,wrong,error,remote,budget,accumulate);});
  auto plan=append_owner_stream(send,dest,fields,error,remote,budget,accumulate);
  require(plan.capacity==capacity&&plan.iterations==(total+capacity-1)/capacity,"packet geometry differs");
  require(plan.reserved_bytes==budget,"packet reservation differs");send.finish();if(peer)receive->finish();
  for(int replay=0;replay<5;++replay) {
    error.fill_(replay==3?19:0);remote.fill_(replay==4?23:0);expected.clear();
    size_t ordinal=0;
    for(size_t i=0;i<fields.size();++i) {
      auto& field=fields[i];const bool live=(i+replay)%3!=0;
      auto value=(at::arange(sizes[i],at::kFloat).remainder(31)+float(replay+1))*.003173f;
      if(replay==2)value.zero_();
      field.source.values.copy_(live?value:at::full_like(value,std::numeric_limits<float>::quiet_NaN()));
      field.source.connected.fill_(live);if(accumulate)field.connected.zero_();
      for(const auto& output:field.destinations) {
        storage[ordinal].fill_(.25f);auto result=storage[ordinal++].cpu();
        if(replay<3&&live) {
          auto x=(accumulate?value+.25f:value).to(output.payload_dtype).to(result.scalar_type());
          result.narrow(1,2,output.values.size(1)).copy_(x.reshape(output.values.sizes()));
        }
        expected.push_back(result);
      }
    }
    portable_torch::synchronize(from);if(peer)portable_torch::synchronize(to);
    send.submit();if(peer)receive->submit();send.wait();if(peer)receive->wait();
    for(size_t i=0;i<storage.size();++i)require(at::equal(storage[i].cpu(),expected[i]),"stream tail/strided/rounding/None result differs");
    for(size_t i=0;accumulate&&i<fields.size();++i)
      require(fields[i].connected.cpu().item<bool>()==(replay<3&&(i+replay)%3!=0),"stream connection differs");
    require(remote.cpu().item<int>()==(replay==3?19:replay==4?23:0),"stream sticky error differs");
  }
  send.close();if(peer)receive->close();if(plan.peer)plan.peer->close();
}
void ordered_reduction(const std::vector<at::Device>& devices) {
  ShardedParameterSources sources;ParameterRegistry registry;
  registry.add("a",at::zeros({513},at::kFloat));registry.add("b",at::zeros({517},at::kFloat));
  registry.add("none",at::zeros({19},at::kFloat));sources.owners=registry.owners();sources.contributions.resize(3);
  const std::vector<float> values{16777216.f,1.f,-16777216.f};
  for(size_t i=0;i<3;++i)for(size_t k=0;k<3;++k) {
    auto f=at::TensorOptions().device(devices[k%devices.size()]).dtype(at::kFloat);
    sources.contributions[i].push_back({at::full(sources.owners[i].value.sizes(),i==2?
      std::numeric_limits<float>::quiet_NaN():values[k],f),at::full({1},i!=2,f.dtype(at::kBool))});
  }
  auto error=at::zeros({1},at::TensorOptions().device(devices[0]).dtype(at::kInt));
  // A budget admitting outputs and small packets, but not full staging copies.
  const int64_t budget=42000;ShardedParameterReduce reduction(sources,devices,error,budget,8*1024*1024);
  require(reduction.stream_reserved_bytes()<budget,"reduction streams exceed tensor budget");
  require(reduction.stream_chunks()>3,"reduction did not exercise multiple packets");reduction.finish();
  for(int replay=0;replay<2;++replay) {
    reduction.run();for(auto e:reduction.errors())require(!e.cpu().item<int>(),"ordered reduction refused");
    for(const auto& part:reduction.gradients()) {
      const auto flags=part.connected.cpu(),result=part.values.cpu();
      for(size_t i=0;i<part.owners.size();++i) {
        require(flags[i].item<bool>()==(part.owners[i].canonical!="none"),"canonical None/zero differs");
        require(at::equal(result.narrow(0,part.offsets[i],part.owners[i].value.numel()),at::zeros_like(part.owners[i].value)),
          "canonical accumulation order changed");
      }
    }
  }
  reduction.close();
}
void calibration(at::Device from,at::Device to) {
  // Caller-owned 16 MiB source/destination exist before the measurement. The
  // transfer must not create another whole-source staging copy on either card.
  const int64_t size=4*1024*1024+3,capacity=65521;
  auto f=at::TensorOptions().device(from).dtype(at::kFloat);
  auto x=at::full({size},.3125f,f),y=at::zeros({size},f.device(to)),on=at::ones({1},f.dtype(at::kBool));
  auto error=at::zeros({1},f.dtype(at::kInt)),remote=at::zeros({1},error.options().device(to));
  std::vector<int64_t> baseline;
  for(auto d:{from,to}) {
    portable_torch::synchronize(d);
    baseline.push_back(c10_npu::NPUCachingAllocator::getDeviceStats(d.index()).allocated_bytes[0].current);
    c10_npu::NPUCachingAllocator::resetPeakStats(d.index());
  }
  CannProgram send(from),receive(to);send.limit_workspace(1024*1024);receive.limit_workspace(1024*1024);
  const auto budget=owner_stream_metadata_bytes(1,1,true)+8*capacity;
  auto plan=append_owner_stream(send,receive,{{{x,on},{{y,at::kFloat}},{}}},error,remote,budget,false);
  send.finish();receive.finish();send.submit();receive.submit();send.wait();receive.wait();
  std::vector<int64_t> peak;size_t i=0;
  for(auto d:{from,to})peak.push_back(c10_npu::NPUCachingAllocator::getDeviceStats(d.index()).allocated_bytes[0].peak-baseline[i++]);
  require(plan.iterations==65,"large stream did not reuse packet");
  require(peak[0]+peak[1]<budget+8*1024*1024,"stream allocator introduced full-sized staging");
  require(at::equal(y.cpu(),x.cpu()),"large streamed publication differs");
  std::cout<<"owner-stream-memory {\"elements\":"<<size<<",\"capacity\":"<<capacity
    <<",\"iterations\":"<<plan.iterations<<",\"reserved_bytes\":"<<plan.reserved_bytes
    <<",\"peak_allocated_delta\":["<<peak[0]<<','<<peak[1]
    <<"],\"cann_workspace_bytes\":["<<send.workspace_bytes()<<','<<receive.workspace_bytes()<<"]}\n";
  send.close();receive.close();plan.peer->close();
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("explicit NPU float32 control test required");
    auto d=portable_torch::resolve_device(args);if(d.type()!=c10::DeviceType::PrivateUse1)throw std::invalid_argument("NPU required");
    at::set_num_threads(1);at::set_num_interop_threads(1);
    {at::NoGradGuard guard;const at::Device remote(d.type(),d.index()+1);int cases=0;
      for(auto to:{d,remote})for(int64_t capacity:{7,255,256,257,513})for(bool reduce:{false,true}) {
        stream_case(d,to,capacity,reduce);++cases;
      }
      ordered_reduction({d,remote});
      calibration(d,remote);
      std::cout<<"owner-stream: passed cases="<<cases<<" replays=100 ordered_reduction=2 None_zero=true strided=true errors=true bounded=true\n";
    }
    runtime.close();return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
