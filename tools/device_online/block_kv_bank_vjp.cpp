#include "block_kv_bank_internal.h"
#include "device_launch_tide_block_kv_bank_numerical_vjp.h"
#include "device_launch_tide_block_kv_bank_snapshot_vjp.h"
namespace tide::device_online {
BlockKVBankGradient append_block_kv_bank_vjp(DeviceProgram& p,const std::shared_ptr<BlockKVBank>& bank,
    const std::vector<BlockKVBankAdjoint>& adjoints,const std::vector<BlockKVBankRawRoot>& raw,
    const at::Tensor& error,int64_t budget){
  using namespace block_bank_detail;if(!bank)throw std::invalid_argument("missing KV bank reverse source");
  const auto& b=*bank;const auto o=b.batch*b.nodes,rows=o*b.length,fw=b.layout.floating_width();
  int64_t ww=0;std::vector<int64_t> table;
  for(const auto& f:b.layout.fields()){table.insert(table.end(),{f.offset,ww,f.width,int64_t(f.ragged)});ww+=f.width;}
  auto borrowed=block_kv_bank_tensors(b);borrowed.push_back(error);
  for(const auto& g:adjoints)borrowed.insert(borrowed.end(),{g.keys,g.values,g.summaries,g.weights,g.snapshot_connected,g.snapshot.mapping,g.snapshot.integer});
  for(const auto& r:raw)borrowed.insert(borrowed.end(),{r.values,r.connected,r.snapshot.mapping,r.snapshot.integer});
  const WeightedStorageIncrement storage(p,std::move(borrowed));admit(o*(4.L*fw+5)+rows*(4.L*ww+5)+4.L*(b.keys.numel()+b.values.numel()+b.summaries.numel()+b.weights.numel())+160,budget);
  const auto options=b.keys.options();auto fields=at::tensor(table,at::kLong).to(error.device());
  BlockKVBankGradient out{at::zeros({o,fw},options),at::zeros({o,5},b.valid.options()),at::zeros({rows,ww},options),at::zeros({rows,5},b.valid.options())};
  auto keys=at::zeros_like(b.keys),values=at::zeros_like(b.values),summaries=at::zeros_like(b.summaries),weights=at::zeros_like(b.weights);
  for(const auto& x:{out.initial,out.initial_connected,out.writes,out.write_connected,keys,values,summaries,weights})p.zero(x);
  for(const auto& a:adjoints){if(a.snapshot.bank.get()!=bank.get())throw std::invalid_argument("KV bank adjoint source identity differs");
    check(a.keys,error,at::kFloat,b.keys.sizes());check(a.values,error,at::kFloat,b.values.sizes());check(a.summaries,error,at::kFloat,b.summaries.sizes());check(a.weights,error,at::kFloat,b.weights.sizes());
    p.add(keys,a.keys);p.add(values,a.values);p.add(summaries,a.summaries);p.add(weights,a.weights);}
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_bank_numerical_vjp)(1,stream,
    ptr(fields),ptr(b.valid),ptr(keys),ptr(values),ptr(summaries),ptr(weights),ptr(out.initial),ptr(out.writes),ptr(error),
    b.batch,b.length,b.nodes,b.capacity,b.entries,fw,ww),"KV unique numerical bank transpose to initial/write blocks");},
    {fields,b.valid,keys,values,summaries,weights,out.initial,out.writes,error});
  auto apply=[&](const BlockKVSnapshot& s,const at::Tensor& values,const at::Tensor& roots,bool dense){
    if(values.defined()!=roots.defined()&&dense)throw std::invalid_argument("KV raw root needs cotangent and mask");
    if(!roots.defined())return;if(s.bank.get()!=bank.get())throw std::invalid_argument("KV snapshot adjoint source identity differs");
    const auto count=o*s.length;check(roots,error,at::kBool,{count,5});if(dense)check(values,error,at::kFloat,{count,fw});
    const auto input=dense?values:out.initial;
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_bank_snapshot_vjp)(1,stream,
      ptr(fields),ptr(b.writes),ptr(s.mapping),ptr(s.integer),ptr(input),ptr(roots),ptr(out.initial),ptr(out.initial_connected),ptr(out.writes),ptr(out.write_connected),ptr(error),
      b.batch,b.length,b.nodes,b.capacity,b.entries,fw,ww,s.length,s.prefix_offset,dense),"KV independent snapshot roots and all causal positive writes");},
      {fields,b.writes,s.mapping,s.integer,input,roots,out.initial,out.initial_connected,out.writes,out.write_connected,error});
  };
  for(const auto& g:adjoints)apply(g.snapshot,{},g.snapshot_connected,false);
  for(const auto& r:raw)apply(r.snapshot,r.values,r.connected,true);
  admit(storage.bytes(),budget);return out;
}
}
