#include "device_backend.h"
#include "peer_exchange.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <future>
#include <iostream>

namespace {
using namespace tide::device_online;
using at::Tensor;
void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
template<class F> void rejects(F f) {
  bool refused=false;try{f();}catch(const std::logic_error&){refused=true;}
  require(refused,"invalid peer layout/capacity/lifecycle accepted");
}
void run_pair(DeviceProgram& a,DeviceProgram& b) {
  // Preserve the constructing thread's contexts. Submit both before waiting;
  // no host observes intermediate scalars or controls an iteration decision.
  a.submit();b.submit();
  rejects([&]{a.submit();});rejects([&]{b.run();});
  std::exception_ptr error;
  try{a.wait();}catch(...){error=std::current_exception();}
  try{b.wait();}catch(...){if(!error)error=std::current_exception();}
  if(error)std::rethrow_exception(error);
  rejects([&]{a.wait();});rejects([&]{b.wait();});
}
int check(at::Device device,at::ScalarType dtype) {
  const at::Device other(device.type(),device.index()+1);at::NoGradGuard guard;
  auto options=at::TensorOptions().device(device).dtype(dtype);
  auto remote=options.device(other),longs=options.dtype(at::kLong),ints=options.dtype(at::kInt),bools=options.dtype(at::kBool);
  // Contiguous views with different nonzero byte offsets exercise raw DMA,
  // descriptor pointers and the retained base-storage lifetimes together.
  auto x=at::zeros({4,5},options).narrow(0,1,3),y=at::zeros({5,5},remote).narrow(0,2,3);
  auto keys=at::zeros({3,6},longs),peer_keys=at::zeros({3,6},longs.device(other));
  auto mask=at::zeros({3},bools),peer_mask=at::zeros({3},bools.device(other));
  auto count=at::zeros({1},longs),peer_count=at::zeros({1},longs.device(other));
  auto command=at::zeros({1},ints),peer_command=at::zeros({1},ints.device(other));
  auto limit=at::zeros({1},longs),iterations=at::zeros({1},longs),budget=at::zeros({1},longs);
  auto unfinished=at::zeros({1},bools),room=at::zeros_like(unfinished),ready=at::zeros_like(unfinished),exhausted=at::zeros({1},ints);
  auto one=at::ones({1},longs),again=at::zeros_like(command),peer_again=at::zeros_like(peer_command);
  auto step=at::ones({1},longs.device(other)),peer_iterations=at::zeros_like(step);
  auto delta=at::full({3,5},.25,remote),key_delta=at::full({3,6},7,longs.device(other));
  auto no=at::zeros({3},bools.device(other)),toggled=at::zeros_like(no);
  PeerExchange::Fields outgoing={{command,peer_command},{count,peer_count},{x,y},{keys,peer_keys},{mask,peer_mask}};
  PeerExchange::Fields incoming={{peer_count,count},{y,x},{peer_keys,keys},{peer_mask,mask}};
  int64_t bytes=0;for(const auto& f:outgoing)bytes+=f.first.nbytes();
  rejects([&]{PeerExchange bad(outgoing,bytes-1);});
  rejects([&]{PeerExchange bad({{x,x}},bytes);});
  rejects([&]{PeerExchange bad({{x,y},{x,y}},bytes*2);});
  rejects([&]{PeerExchange bad({{x.transpose(0,1),y.transpose(0,1)}},bytes);});
  rejects([&]{PeerExchange bad({{x,y.to(at::kLong)}},bytes);});
  rejects([&]{PeerExchange bad({},bytes);});
  rejects([&]{PeerExchange bad(outgoing,bytes,0);});
  PeerExchange send(outgoing,bytes),receive(incoming,bytes);
  require(send.packet_bytes()==bytes,"peer packet byte accounting differs");
  DeviceProgram source(device),destination(other);
  rejects([&]{send.append_send(destination);});
  rejects([&]{send.append_receive(source);});
  const auto head=source.label(),body=source.label(),end=source.label();
  const auto peer_head=destination.label(),peer_body=destination.label(),peer_end=destination.label();
  source.mark(head);source.less(count,limit,unfinished);source.less(iterations,budget,room);
  source.logical_and(unfinished,room,ready);source.cast_index(ready,command);
  send.append_send(source);source.branch(command,{end,body});
  source.mark(body);receive.append_receive(source);source.add(iterations,one);source.branch(again,{head});
  source.mark(end);source.less(count,limit,unfinished);source.cast_index(unfinished,exhausted);
  destination.mark(peer_head);send.append_receive(destination);destination.branch(peer_command,{peer_end,peer_body});
  destination.mark(peer_body);destination.add(peer_count,step);destination.add(y,delta);destination.add(peer_keys,key_delta);
  destination.equal(peer_mask,no,toggled);destination.copy(peer_mask,toggled);
  destination.add(peer_iterations,at::ones_like(step));receive.append_send(destination);
  destination.branch(peer_again,{peer_head});destination.mark(peer_end);
  source.finish();destination.finish();
  // A fresh host thread must fail before submitting a peer-dependent model.
  std::async(std::launch::async,[&]{rejects([&]{source.submit();});}).get();
  rejects([&]{source.wait();});rejects([&]{source.run(0);});
  rejects([&]{send.close();});rejects([&]{receive.close();});
  const auto original=at::arange(15,at::kFloat).reshape({3,5}).mul(.03125).to(dtype);
  auto expected=original.clone();auto expected_keys=at::arange(18,at::kLong).reshape({3,6})+((int64_t(1)<<55)+17);
  auto expected_mask=at::tensor({1,0,1},at::kLong).to(at::kBool);int64_t cpu_count=0;int cases=0;
  x.copy_(expected);keys.copy_(expected_keys);mask.copy_(expected_mask);
  auto run=[&](int64_t target,int64_t increment,int64_t capacity) {
    limit.fill_(target);step.fill_(increment);budget.fill_(capacity);iterations.zero_();peer_iterations.zero_();
    int64_t n=0;
    while(cpu_count<target&&n<capacity){cpu_count+=increment;expected=(expected.to(at::kFloat)+.25).to(dtype);expected_keys+=7;expected_mask=expected_mask.logical_not();++n;}
    portable_torch::synchronize(device);portable_torch::synchronize(other);run_pair(source,destination);
    require(count.cpu().item<int64_t>()==cpu_count,"device peer loop count differs from independent CPU");
    require(iterations.cpu().item<int64_t>()==n&&peer_iterations.cpu().item<int64_t>()==n,"peer executed wrong number of actions");
    require(exhausted.cpu().item<int32_t>()==int(cpu_count<target),"peer capacity status differs");
    require(at::equal(x.cpu(),expected),"peer payload/buffer reuse corrupted");
    require(at::equal(keys.cpu(),expected_keys),"peer int64 identity lost precision");
    require(at::equal(mask.cpu(),expected_mask),"peer bool mask changed");++cases;
  };
  run(5,1,0);run(5,1,16);run(5,1,16);run(19,3,2);run(19,3,4);
  run(29,3,2);run(29,3,16);run(158,1,256);run(158,1,256);
  cpu_count=(int64_t(1)<<55)+17;count.fill_(cpu_count);
  run(cpu_count+7,2,2);run(cpu_count+7,2,16);
  source.close();destination.close();send.close();receive.close();send.close();
  rejects([&]{send.append_send(source);});
  std::cout<<"device-peer: passed cases="<<cases<<" dtype="<<portable_torch::dtype_name(dtype)
    <<" dynamic_iterations=true exact_int64=true notification_channels=4 packet_bytes="<<bytes
    <<" host_decisions_per_iteration=0 scope=transport_component\n";
  return cases;
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))
      throw std::invalid_argument("peer check requires explicit NPU FP32/FP16 and two visible devices");
    args.allow_npu_float16=true;const auto device=portable_torch::resolve_device(args);
    if(device.type()!=tide::device_online::resident_device_type||device.index()!=0)
      throw std::invalid_argument("peer gate expects remapped logical npu:0 and npu:1");
    at::set_num_threads(1);at::set_num_interop_threads(1);check(device,args.dtype);runtime.close();return 0;
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
