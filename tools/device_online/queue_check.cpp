#include "packed_queue.h"
#include "queue_closure.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

namespace {
using namespace tide::device_online;
using I=int64_t;
void require(bool ok,const char* why) { if(!ok)throw std::runtime_error(why); }
AtomBatch batch(const std::vector<I>& rows,const std::vector<float>& values,at::Device device,at::ScalarType dtype) {
  I count=rows.size()/6,width=values.size()/count;
  return {at::tensor(rows,at::kLong).reshape({count,6}).to(device),
          at::tensor(values,at::kFloat).reshape({count,width}).to(device,dtype),
          at::ones({count},at::TensorOptions().dtype(at::kBool).device(device))};
}
void queue_cases(at::Device device,at::ScalarType dtype) {
  PackedQueue queue(3,2,at::TensorOptions().device(device).dtype(dtype));
  const I big=(I(1)<<55)+3;
  auto two=batch({0,1,big,1,8,big-2, 0,1,big,1,7,big-3},{0,0,2,3},device,dtype);
  queue.append(two);
  auto one=batch({0,1,big,1,9,big-1},{4,5},device,dtype);
  queue.append(one);
  require(queue.size().cpu().item<I>()==3,"zero-valued message lost its identity");
  auto packed=queue.pack(queue.atoms().valid);
  require(at::equal(packed.coordinates.select(1,4).cpu(),at::tensor({7,8,9},at::kLong)),"unstable physical-edge packing");
  require((packed.coordinates.select(1,2)==big).all().cpu().item<bool>(),"int64 packing lost time precision");
  auto before=queue.atoms().coordinates.clone(),payload=queue.atoms().values.clone();
  queue.append(one);
  require(queue.error().cpu().item<int>()==1,"queue overflow was not reported");
  require(at::equal(before,queue.atoms().coordinates)&&at::equal(payload,queue.atoms().values),"overflow partially mutated queue");
  queue.erase(queue.atoms().valid);
  require(queue.size().cpu().item<I>()==3,"sticky failure allowed queue mutation");
  queue.clear();
  for(int j=0;j<8;++j) {
    queue.append(two);
    queue.replace(queue.atoms().valid,one);
    require(queue.size().cpu().item<I>()==1,"consumed storage did not recycle");
    queue.erase(queue.atoms().valid);
  }
  require(queue.peak().cpu().item<I>()==2,"capacity counted cumulative rather than live messages");
  queue.clear();queue.append(two);
  auto all=queue.atoms().valid.clone();
  AtomBatch too_many{at::cat({two.coordinates,two.coordinates}),at::cat({two.values,two.values}),
                    at::cat({two.valid,two.valid})};
  before=queue.atoms().coordinates.clone();
  queue.replace(all,too_many);
  require(queue.error().cpu().item<int>()==1&&queue.size().cpu().item<I>()==2
          &&at::equal(before,queue.atoms().coordinates),"replace overflow consumed inputs before refusal");
}

// Independent scalar oracle enumerates possible unfinished region arrivals only
// in these tiny validation horizons. Production uses no potential-event table.
std::vector<I> expected(const std::vector<I>& owners,I regions,const std::vector<Wire>& wires,
                        const std::vector<I>& rows,I stop,bool prefill) {
  std::vector<I> safe(2*regions,stop),earliest(2,stop),answer;
  std::set<std::tuple<I,I,I>> future;
  for(size_t i=0;i<rows.size();i+=6) {
    I b=rows[i],node=rows[i+1],time=rows[i+2];if(time>=stop)continue;
    earliest[b]=std::min(earliest[b],time);
    for(auto [s,t,d]:wires)if(owners[s]==owners[node]&&d<stop-time)
      future.insert({time+d,b,owners[t]});
  }
  while(!future.empty()) {
    auto [time,b,r]=*future.begin();future.erase(future.begin());safe[b*regions+r]=std::min(safe[b*regions+r],time);
    for(auto [s,t,d]:wires)if(owners[s]==r&&d<stop-time)future.insert({time+d,b,owners[t]});
  }
  for(size_t i=0;i<rows.size();i+=6) {
    I b=rows[i],node=rows[i+1],time=rows[i+2];
    answer.push_back(time<stop&&(prefill?time<safe[b*regions+owners[node]]:time==earliest[b]));
  }
  return answer;
}
void closure_cases(at::Device device,at::ScalarType dtype) {
  const std::vector<I> owners{0,1,1,2};
  const std::vector<std::vector<Wire>> graphs{
    {}, {{0,1,1},{2,3,3},{0,1,1}}, {{0,1,2},{1,0,3},{2,3,1},{3,2,2}},
    {{0,0,1},{0,2,3},{2,1,2},{3,0,1}},
    {{0,1,std::numeric_limits<I>::max()},{1,3,2},{3,0,3}}};
  for(const auto& wires:graphs)for(I base:{I(0),(I(1)<<55)+17}) {
    std::vector<I> rows;std::vector<float> values;
    for(I b=0;b<2;++b)for(I n=0;n<4;++n)for(I j=0;j<2;++j) {
      rows.insert(rows.end(),{b,n,base+(n*3+j*5+b)%9,0,n,j});values.push_back(float(j));
    }
    auto atoms=batch(rows,values,device,dtype);
    auto stop=at::full({1},base+8,at::TensorOptions().device(device).dtype(at::kLong));
    // The small explicit budget forces two sample chunks, versus one below.
    QueueClosure split(owners,3,wires,2,device,500),full(owners,3,wires,2,device,4096);
    require(split.sample_chunk()==1&&full.sample_chunk()==2,"closure budget did not change physical chunks");
    for(bool prefill:{false,true}) {
      auto oracle=at::tensor(expected(owners,3,wires,rows,base+8,prefill),at::kLong).to(at::kBool);
      require(at::equal(split.ready(atoms,stop,prefill).cpu(),oracle),"tensor closure disagrees with independent arrival oracle");
      require(at::equal(full.ready(atoms,stop,prefill).cpu(),oracle),"closure chunking changed logical ready set");
    }
  }
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto")throw std::invalid_argument("queue check requires explicit device");
    auto device=portable_torch::resolve_device(args);
    at::set_num_threads(1);at::set_num_interop_threads(1);at::NoGradGuard guard;
    queue_cases(device,args.dtype);
    if(device.is_cpu())closure_cases(device,args.dtype);
    portable_torch::synchronize(device);
    std::cout<<"packed-queue: passed physical_edges=true zero_message=true atomic_capacity=true recycling=true"
             <<" closure_oracle_cases="<<(device.is_cpu()?40:0)
             <<" exact_int64=true closure_scope="<<(device.is_cpu()?"cpu-reference":"separate-ascendc-check")<<'\n';
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
