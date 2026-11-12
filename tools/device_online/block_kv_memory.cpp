#include "block_kv_memory.h"
#include "block_kv_bank_internal.h"
#include "weighted_saved_tensors.h"
namespace tide::device_online {
using namespace block_bank_detail;
std::vector<at::Tensor> block_kv_memory_tensors(const BlockKVMemoryTape& t){
  auto out=block_kv_read_plan_tensors(t.plan);out.insert(out.end(),{t.memory,t.total});
  for(const auto* projection:{&t.query,&t.key,&t.value})if(*projection)append_saved_tensors(out,**projection);
  if(t.match){auto a=block_kv_match_tensors(*t.match);out.insert(out.end(),a.begin(),a.end());}
  for(const auto& pool:t.pools){auto a=block_kv_pool_tensors(pool);out.insert(out.end(),a.begin(),a.end());}return out;
}
BlockKVMemoryTape append_block_kv_memory(DeviceProgram& p,const weighted::KVConfig& c,const BlockKVSnapshot& s,
    const at::Tensor& times,const at::Tensor& input,const at::Tensor& owners,const at::Tensor& present,
    const WeightedKVQueryParameters& w,const at::Tensor& error,int64_t budget){
  c.validate();if(!s.bank)throw std::invalid_argument("KV bank Memory snapshot missing");
  const auto& b=*s.bank;const auto o=b.batch*b.nodes,rows=o*s.length,k=b.entries,d=c.width,ds=c.summary_width;const bool qry=c.memory_read=="qry";
  if(qry&&(!w.query.defined()||w.query.dim()!=3))throw std::invalid_argument("KV bank query parameters missing");
  const auto banks=qry?w.query.size(0):1;
  if(qry)for(const auto& x:{w.query,w.key,w.value})check(x,error,at::kFloat,{banks,ds,d});
  auto borrowed=block_kv_bank_tensors(b);borrowed.insert(borrowed.end(),{s.mapping,s.integer,times,input,owners,present,w.query,w.key,w.value,error});
  const WeightedStorageIncrement storage(p,std::move(borrowed));auto remaining=[&](){admit(storage.bytes(),budget);return budget-storage.bytes();};
  BlockKVMemoryTape t;t.plan=append_block_kv_read_plan(p,c,s,times,input,owners,present,banks,qry,false,error,remaining());
  at::Tensor values=b.summaries,logits;
  if(qry){t.query=append_weighted_projection(p,t.plan.clean,t.plan.query_owners,{w.query,{}},c.epsilon,error,remaining());
    t.key=append_weighted_projection(p,b.keys.reshape({o*k,d}),t.plan.entry_owners,{w.key,{}},std::nullopt,error,remaining());
    t.value=append_weighted_projection(p,b.values.reshape({o*k,d}),t.plan.entry_owners,{w.value,{}},std::nullopt,error,remaining());
    t.match=append_block_kv_match(p,owner_order(p,s,t.query->output),t.key->output.reshape({o,k,ds}),t.plan.ages,t.plan.visible,c.position_scale,c.read_frequencies,0,error,remaining());
    values=t.value->output.reshape({o,k,ds});logits=t.match->logits;
  }
  auto memories=at::zeros({o,s.length,2*ds},input.options()),totals=at::zeros({o,s.length,2},input.options());
  for(int64_t channel=0;channel<2;++channel){auto pool=append_block_kv_pool(p,t.plan.weights[channel],values,logits,t.plan.visible,t.plan.present,error,remaining());
    put(p,memories,channel*ds,pool.output);put(p,totals,channel,pool.total.unsqueeze(2));t.pools.push_back(std::move(pool));}
  t.memory=row_order(p,s,memories).reshape({rows,2,ds});t.total=row_order(p,s,totals);admit(storage.bytes(),budget);return t;
}
}
