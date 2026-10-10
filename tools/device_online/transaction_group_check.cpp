#include "queue_transaction.h"
#include "portable_torch/runtime.hpp"
#include <ATen/core/grad_mode.h>
#include <stdexcept>
#include <iostream>

namespace {
using namespace tide::device_online;
using I=int64_t;
AtomBatch copy_cpu(const AtomBatch& q){return {q.coordinates.cpu().clone(),q.values.cpu().clone(),q.valid.cpu().clone()};}
void unchanged(const AtomBatch& actual,const AtomBatch& expected) {
  if(!at::equal(actual.coordinates.cpu(),expected.coordinates)||!at::equal(actual.values.cpu(),expected.values)
      ||!at::equal(actual.valid.cpu(),expected.valid))throw std::runtime_error("refused transaction group modified a queue");
}
void live_equal(const AtomBatch& actual,const AtomBatch& expected) {
  auto valid=actual.valid.cpu();
  if(valid.sum().item<I>()!=expected.valid.sum().item<I>()
      ||!at::equal(actual.coordinates.cpu().index({valid}),expected.coordinates.index({expected.valid}))
      ||!at::equal(actual.values.cpu().index({valid}),expected.values.index({expected.valid})))
    throw std::runtime_error("cross-queue snapshot read overwritten payloads");
}
}
void check_transaction_group(at::Device device,at::ScalarType dtype) {
  at::NoGradGuard guard;auto opts=at::TensorOptions().device(device).dtype(dtype);
  auto error=at::zeros({1},opts.dtype(at::kInt));
  QueueTransaction a(5,3,2,1,opts,error),b(4,3,2,1,opts,error);
  auto ca=at::ones({5},error.options()),cb=at::ones({4},error.options());
  DeviceProgram program(device);
  auto first=a.propose_stage(program,ca,b.atoms()),second=b.propose_stage(program,cb,a.atoms());
  a.commit_stage(program,first);b.commit_stage(program,second);program.finish();
  auto seed=[](QueueTransaction& q,I count,I base) {
    auto rows=q.atoms().valid.numel();std::vector<I> c;
    for(I i=0;i<rows;++i)c.insert(c.end(),{0,i%2,(I(1)<<55)+base+i,1,i,base});
    q.atoms().coordinates.copy_(at::tensor(c,at::kLong).reshape({rows,6}));
    q.atoms().values.copy_(at::arange(rows*3,at::kFloat).reshape({rows,3})+base);
    q.atoms().valid.copy_(at::arange(rows,at::kLong)<count);q.stats().fill_(count);
  };
  seed(a,3,10);seed(b,2,30);
  for(int round=0;round<4;++round) {
    auto old_a=copy_cpu(a.atoms()),old_b=copy_cpu(b.atoms());
    portable_torch::synchronize(device);program.run();
    if(error.cpu().item<int>()!=0)throw std::runtime_error("legal cross-queue swap refused");
    live_equal(a.atoms(),old_b);live_equal(b.atoms(),old_a);
  }
  // A's proposal fits, B's does not. Neither queue may commit. This also
  // guards outputs/state against a later pending-message capacity failure.
  for(bool replace:{false,true}) {
    error.zero_();seed(a,replace?5:3,40);seed(b,2,70);ca.fill_(replace);cb.fill_(replace);
    auto old_a=copy_cpu(a.atoms()),old_b=copy_cpu(b.atoms());auto sa=a.stats().cpu(),sb=b.stats().cpu();
    portable_torch::synchronize(device);program.run();
    if(error.cpu().item<int>()!=1)throw std::runtime_error("group capacity refusal missing");
    unchanged(a.atoms(),old_a);unchanged(b.atoms(),old_b);
    if(!at::equal(sa,a.stats().cpu())||!at::equal(sb,b.stats().cpu()))throw std::runtime_error("refused group changed counts");
  }
  program.close();std::cout<<"device-queue-group: passed cases=6 cross_queue_aliases=true atomic_refusal=true\n";
}
