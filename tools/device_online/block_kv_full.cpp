#include "block_kv_full.h"
#include "block_kv_bank_internal.h"
#include "weighted_saved_tensors.h"
namespace tide::device_online {
using namespace block_bank_detail;
std::vector<at::Tensor> block_kv_full_tensors(const BlockKVFullTape& t){
  auto out=block_kv_read_plan_tensors(t.plan);append_saved_tensors(out,t.query);append_saved_tensors(out,t.residual);out.push_back(t.output);
  for(const auto& m:t.matches){auto a=block_kv_match_tensors(m);out.insert(out.end(),a.begin(),a.end());}
  for(const auto& pool:t.pools){auto a=block_kv_pool_tensors(pool);out.insert(out.end(),a.begin(),a.end());}return out;
}
BlockKVFullTape append_block_kv_full(DeviceProgram& p,const weighted::KVConfig& c,const BlockKVSnapshot& s,
    const at::Tensor& times,const at::Tensor& input,const at::Tensor& owners,const at::Tensor& active,
    const WeightedKVFullParameters& w,const at::Tensor& error,int64_t budget){
  c.validate();if(!s.bank||!w.query.defined()||w.query.dim()!=3)throw std::invalid_argument("KV bank Full snapshot/parameter bank missing");
  const auto& b=*s.bank;const auto o=b.batch*b.nodes,q=s.length,k=b.entries,d=c.width,h=c.heads,dh=d/h,banks=w.query.size(0);
  check(w.query,error,at::kFloat,{banks,d,d});check(w.residual.out,error,at::kFloat,{banks,d,d});
  check(w.residual.gate,error,at::kFloat,{banks,c.ffn_width,d});check(w.residual.up,error,at::kFloat,{banks,c.ffn_width,d});
  check(w.residual.down,error,at::kFloat,{banks,d,c.ffn_width});
  auto borrowed=block_kv_bank_tensors(b);borrowed.insert(borrowed.end(),{s.mapping,s.integer,times,input,owners,active,w.query,w.residual.out,w.residual.gate,w.residual.up,w.residual.down,error});
  const WeightedStorageIncrement storage(p,std::move(borrowed));auto remaining=[&](){admit(storage.bytes(),budget);return budget-storage.bytes();};
  BlockKVFullTape t;t.plan=append_block_kv_read_plan(p,c,s,times,input,owners,active,banks,true,true,error,remaining());
  t.query=append_weighted_projection(p,t.plan.clean,t.plan.query_owners,{w.query,{}},c.epsilon,error,remaining());
  auto query=owner_order(p,s,t.query.output),memory=at::zeros({o,q,d},input.options());
  for(int64_t head=0;head<h;++head){
    auto match=append_block_kv_match(p,columns(p,query,head*dh,dh),columns(p,b.keys,head*dh,dh),t.plan.ages,t.plan.visible,
      c.position_scale,c.full_frequencies.empty()?std::vector<double>{}:c.full_frequencies[head],
      c.distance_slopes.empty()?0:c.distance_slopes[head],error,remaining());
    auto pool=append_block_kv_pool(p,t.plan.weights[1],columns(p,b.values,head*dh,dh),match.logits,t.plan.visible,t.plan.present,error,remaining());
    put(p,memory,head*dh,pool.output);t.matches.push_back(std::move(match));t.pools.push_back(std::move(pool));
  }
  t.residual=append_weighted_residual(p,input,row_order(p,s,memory),t.plan.full_owners,w.residual,c.epsilon,c.alpha_att,c.alpha_ff,error,remaining());
  t.output=t.residual.output;admit(storage.bytes(),budget);return t;
}
BlockKVFullGradient append_block_kv_full_vjp(DeviceProgram& p,const BlockKVFullTape& t,const at::Tensor& gradient,
    const at::Tensor& roots,const at::Tensor& error,int64_t budget){
  const auto& s=t.plan.snapshot;const auto& b=*s.bank;const auto o=b.batch*b.nodes,q=s.length,k=b.entries,rows=o*q,d=t.output.size(1),h=int64_t(t.pools.size()),dh=d/h;
  check(gradient,error,at::kFloat,{rows,d});check(roots,error,at::kBool,{rows});
  auto borrowed=block_kv_full_tensors(t);borrowed.insert(borrowed.end(),{gradient,roots,error});
  const WeightedStorageIncrement storage(p,std::move(borrowed));auto remaining=[&](){admit(storage.bytes(),budget);return budget-storage.bytes();};
  auto residual=append_weighted_residual_vjp(p,t.residual,gradient,roots,error,remaining());
  auto bar=owner_order(p,s,residual.memory),connected=owner_order(p,s,residual.connected),pool_roots=at::zeros({o,q,2},roots.options());put(p,pool_roots,0,connected);
  auto flags=at::zeros({h,o,q,3},roots.options()),weight_bars=at::zeros_like(t.plan.weights),query_bar=at::zeros({o,q,d},gradient.options());
  auto key_bar=at::zeros_like(b.keys),value_bar=at::zeros_like(b.values);p.zero(weight_bars);
  for(int64_t head=0;head<h;++head){auto pool=append_block_kv_pool_vjp(p,t.pools[head],columns(p,bar,head*dh,dh),at::zeros({o,q},gradient.options()),pool_roots,error,remaining());
    auto match=append_block_kv_match_vjp(p,t.matches[head],pool.logits,error,remaining());p.copy(flags[head],pool.connected);
    put(p,query_bar,head*dh,match.query);put(p,key_bar,head*dh,match.keys);put(p,value_bar,head*dh,pool.values);p.add(weight_bars[1],pool.weights);}
  auto dependencies=append_block_kv_read_roots(p,t.plan,flags,2,error);
  auto query=append_weighted_projection_vjp(p,t.query,row_order(p,s,query_bar),dependencies.query,error,remaining());
  auto input=at::zeros_like(gradient);p.copy(input,residual.input);p.add(input,query.input);
  BlockKVFullGradient out;
  out.bank={key_bar,value_bar,at::zeros_like(b.summaries),append_block_kv_weight_vjp(p,t.plan,weight_bars,error),dependencies.snapshot,s};
  out.rows={{},{},input,residual.connected,{query.parameters.matrix,residual.parameters},query.owner_connected,
    residual.out_connected,residual.gate_connected,residual.up_connected,residual.down_connected};
  admit(storage.bytes(),budget);return out;
}
}
