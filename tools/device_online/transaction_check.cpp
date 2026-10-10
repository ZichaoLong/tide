#include "device_backend.h"
#include "queue_transaction.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <iostream>
#include <limits>

void check_transaction_group(at::Device,at::ScalarType);

namespace {
using namespace tide::device_online;
using I=int64_t;
void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
void unchanged(const AtomBatch& before,const AtomBatch& after) {
  require(at::equal(before.coordinates,after.coordinates.cpu())&&at::equal(before.valid,after.valid.cpu())
      &&at::equal(before.values,after.values.cpu()),"failed transaction modified queue");
}
void check(at::Device device,at::ScalarType dtype) {
  at::NoGradGuard guard;
  const I capacity=11,arrivals=5,nodes=4,samples=2,width=7;
  const auto opts=at::TensorOptions().device(device).dtype(dtype),cpu=opts.device(at::kCPU);
  QueueTransaction queue(capacity,width,nodes,samples,opts);
  PackedQueue oracle(capacity,width,cpu);
  AtomBatch incoming{at::zeros({arrivals,6},opts.dtype(at::kLong)),at::zeros({arrivals,width},opts),
                     at::zeros({arrivals},opts.dtype(at::kBool))};
  auto consumed=at::zeros({capacity},opts.dtype(at::kInt));
  DeviceProgram program(device);queue.append_stage(program,consumed,incoming);program.finish();
  I cases=0,emitted=0;
  // Reuse a single program with different exact times, masks and payloads.
  // The CPU oracle uses stable tensor sorting; it does not call the AIV planner.
  for(I round=0;round<40;++round) {
    auto remove=(at::arange(capacity,at::kLong)+round).remainder(3)!=0;
    std::vector<I> rows;
    for(I i=0;i<arrivals;++i)rows.insert(rows.end(),{i%samples,(i+round)%nodes,
      (I(1)<<55)+round*7+i,1,i,round});
    // Two physically distinct parallel arrivals, and a present numerical zero.
    rows[6]=rows[0];rows[7]=rows[1];rows[8]=rows[2];rows[11]=rows[5];
    auto coords=at::tensor(rows,at::kLong).reshape({arrivals,6});
    auto values=(at::arange(arrivals*width,at::kFloat).reshape({arrivals,width})+round).to(dtype);
    values[0].zero_();
    auto present=at::arange(arrivals,at::kLong).remainder(4)!=round%4;
    auto masked=~present;
    coords.index_put_({masked},-99);values.index_put_({masked},std::numeric_limits<double>::quiet_NaN());
    AtomBatch input{coords,values,present};
    oracle.replace(remove,input);require(oracle.error().item<int>()==0,"test exceeded CPU capacity");
    incoming.coordinates.copy_(coords);incoming.values.copy_(values);incoming.valid.copy_(present);
    consumed.copy_(remove.to(at::kInt));portable_torch::synchronize(device);program.run();
    const auto got=queue.atoms(),expected=oracle.atoms();auto live=got.valid.cpu();
    require(queue.error().cpu().item<int>()==0,"device rejected valid queue transaction");
    require(at::equal(live,expected.valid),"device queue live bits differ");
    require(at::equal(got.coordinates.cpu().index({live}),expected.coordinates.index({live})),"device queue exact identity/order differs");
    require(at::equal(got.values.cpu().index({live}),expected.values.index({live})),"bulk payload placement differs");
    require(got.values.cpu().index({~live}).eq(0).all().item<bool>(),"invalid payload leaked into padding");
    auto stats=queue.stats().cpu();
    require(stats[0].item<I>()==oracle.size().item<I>()&&stats[1].item<I>()==oracle.peak().item<I>(),"live capacity statistics differ");
    emitted+=present.sum().item<I>();++cases;
  }
  require(emitted>capacity*8,"capacity recycling was not exercised");
  // Empty transaction drops consumed atoms, preserving maximum occupancy.
  consumed.fill_(1);incoming.valid.zero_();portable_torch::synchronize(device);program.run();
  require(queue.stats().cpu()[0].item<I>()==0&&!queue.atoms().valid.cpu().any().item<bool>(),"empty queue transition failed");
  ++cases;
  // Errors must preserve every old slot (including holes), counts and peak.
  // Reset only at test boundaries; production errors require caller recovery.
  for(int failure=0;failure<5;++failure) {
    queue.error().zero_();queue.atoms().coordinates.zero_();queue.atoms().values.fill_(17);
    queue.atoms().valid.fill_(true);queue.stats().fill_(capacity);
    consumed.zero_();incoming.coordinates.zero_();incoming.values.fill_(23);incoming.valid.fill_(true);
    const int expected=failure==0?1:2;
    if(failure==1)incoming.coordinates.select(1,1).fill_(nodes);
    if(failure==2)incoming.coordinates.select(1,2).fill_(-1);
    if(failure==3)consumed.fill_(2);
    if(failure==4)queue.atoms().coordinates.select(1,0).fill_(samples);
    AtomBatch before{queue.atoms().coordinates.cpu(),queue.atoms().values.cpu(),queue.atoms().valid.cpu()};
    auto stats=queue.stats().cpu();portable_torch::synchronize(device);program.run();
    require(queue.error().cpu().item<int>()==expected,"device queue error classification differs");
    unchanged(before,queue.atoms());require(at::equal(stats,queue.stats().cpu()),"refused transaction changed counters");
    // Correcting inputs does not silently clear an earlier failure.
    consumed.fill_(1);incoming.valid.zero_();portable_torch::synchronize(device);program.run();
    require(queue.error().cpu().item<int>()==expected,"device queue error was not sticky");
    unchanged(before,queue.atoms());++cases;
  }
  program.close();
  std::cout<<"device-queue: passed cases="<<cases<<" payload_dtype="<<c10::toString(dtype)
    <<" exact_int64=true zero_message=true atomic_capacity=true recycling=true host_decisions_per_transaction=0\n";
}
}
int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;
  try {
    auto args=portable_torch::parse_cli(argc,argv,true);
    if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||(args.dtype!=at::kFloat&&args.dtype!=at::kHalf))
      throw std::invalid_argument("device queue check requires explicit NPU and FP32/FP16");
    args.allow_npu_float16=true;auto device=portable_torch::resolve_device(args);
    if(device.type()!=tide::device_online::resident_device_type)throw std::invalid_argument("device queue check requires NPU");
    at::set_num_threads(1);at::set_num_interop_threads(1);check(device,args.dtype);
    check_transaction_group(device,args.dtype);return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
