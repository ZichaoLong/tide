#include "block_kv_read_plan.h"
#include "block_kv_bank_internal.h"
#include "device_launch_tide_block_kv_read_plan.h"
#include <cmath>
namespace tide::device_online {
std::vector<at::Tensor> block_kv_read_plan_tensors(const BlockKVReadPlan& t){
  auto out=block_kv_bank_tensors(*t.snapshot.bank);
  out.insert(out.end(),{t.snapshot.mapping,t.snapshot.integer,t.weights,t.decay,t.visible,t.ages,t.present,t.query_owners,t.entry_owners,t.full_owners,t.clean});return out;
}
BlockKVReadPlan append_block_kv_read_plan(DeviceProgram& p,const weighted::KVConfig& c,const BlockKVSnapshot& s,
    const at::Tensor& times,const at::Tensor& input,const at::Tensor& owners,const at::Tensor& present,
    int64_t banks,bool query,bool full,const at::Tensor& error,int64_t budget){
  using namespace block_bank_detail;
  if(!s.bank||s.length<1||banks<1||!std::isfinite(float(c.beta_short))||float(c.beta_short)<=0)
    throw std::invalid_argument("missing KV bank snapshot/owners or unrepresentable decay");
  const auto& b=*s.bank;const auto o=b.batch*b.nodes,rows=o*s.length,k=b.entries,d=c.width;
  check(times,error,at::kLong,{rows});check(input,error,at::kFloat,{rows,d});check(owners,error,at::kLong,{rows});
  check(present,error,at::kBool,{rows});check(s.mapping,error,at::kLong,{rows,b.capacity});check(s.integer,error,at::kLong,{rows,b.capacity+4});
  auto borrowed=block_kv_bank_tensors(b);borrowed.insert(borrowed.end(),{s.mapping,s.integer,times,input,owners,present,error});
  const WeightedStorageIncrement storage(p,std::move(borrowed));admit(rows*(25.L*k+25+4.L*d)+8.L*o*k,budget);
  BlockKVReadPlan t;t.snapshot=s;t.weights=at::zeros({2,o,s.length,k},input.options());t.decay=at::zeros_like(t.weights);
  t.visible=at::zeros({o,s.length,k},present.options());t.ages=at::zeros({o,s.length,k},times.options());t.present=at::zeros({o,s.length},present.options());
  t.query_owners=at::full_like(owners,-1);t.entry_owners=at::full({o*k},-1,owners.options());t.full_owners=at::full_like(owners,-1);t.clean=at::zeros_like(input);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_read_plan)(1,stream,
    ptr(s.mapping),ptr(s.integer),ptr(b.weights),ptr(b.times),ptr(times),ptr(input),ptr(owners),ptr(present),
    ptr(t.weights),ptr(t.decay),ptr(t.visible),ptr(t.ages),ptr(t.present),ptr(t.query_owners),ptr(t.entry_owners),ptr(t.full_owners),ptr(t.clean),ptr(error),
    b.batch,s.length,b.nodes,b.capacity,k,d,banks,c.window,query,full,float(c.beta_short)),"actual KV snapshot incidence and projection-use union");},
    {s.mapping,s.integer,b.weights,b.times,times,input,owners,present,t.weights,t.decay,t.visible,t.ages,t.present,t.query_owners,t.entry_owners,t.full_owners,t.clean,error});
  admit(storage.bytes(),budget);return t;
}
}
