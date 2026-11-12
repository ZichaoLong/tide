#include "block_kv_bank.h"
#include "weighted_storage_budget.h"
#include "device_launch_tide_block_kv_bank_plan.h"
#include "device_launch_tide_block_kv_bank_decode.h"
#include <limits>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void check(const at::Tensor& x,const at::Tensor& e,at::ScalarType type,at::IntArrayRef shape){
  if(!x.defined()||x.device()!=e.device()||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid KV bank tensor");
}
void admit(long double bytes,int64_t budget){if(budget<1||bytes>budget)throw std::invalid_argument("KV bank tensor budget exceeded");}
at::Tensor fields(const WeightedRawLayout& layout,const at::Tensor& like){
  std::vector<int64_t> values;int64_t offset=0;
  for(const auto& f:layout.fields()){values.insert(values.end(),{f.offset,offset,f.width,int64_t(f.ragged)});offset+=f.width;}
  return at::tensor(values,at::kLong).to(like.device());
}
}
std::vector<at::Tensor> block_kv_bank_tensors(const BlockKVBank& b){
  return {b.initial.floating,b.initial.integer,b.keys,b.values,b.summaries,b.weights,b.times,b.valid,
    b.old_mapping,b.proposal_mapping,b.old_integer,b.proposal_integer,b.writes,b.final_mapping};
}
BlockKVSnapshot block_kv_snapshot(const BlockKVSequenceTape& t,bool proposal){
  if(!t.bank)throw std::invalid_argument("KV bank snapshot requires a bank sequence");
  return {t.bank,proposal?t.bank->proposal_mapping:t.bank->old_mapping,
    proposal?t.bank->proposal_integer:t.bank->old_integer,t.length,proposal?1:0};
}
BlockKVSnapshot block_kv_final_snapshot(const BlockKVSequenceTape& t){
  if(!t.bank)throw std::invalid_argument("KV bank final snapshot requires a bank");
  return {t.bank,t.bank->final_mapping,t.final.integer,1,t.length};
}
WeightedRawState append_block_kv_bank_decode(DeviceProgram& p,const BlockKVSnapshot& s,const at::Tensor& error,int64_t budget){
  if(!s.bank||s.length<1)throw std::invalid_argument("missing KV bank decode snapshot");
  const auto& b=*s.bank;const auto rows=b.batch*s.length*b.nodes,fw=b.layout.floating_width();
  check(s.mapping,error,at::kLong,{rows,b.capacity});check(s.integer,error,at::kLong,{rows,4+b.capacity});
  admit(4.L*rows*fw+160,budget);auto table=fields(b.layout,error);
  auto raw=at::zeros({rows,fw},b.keys.options());
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_bank_decode)(1,stream,
    ptr(table),ptr(b.initial.floating),ptr(b.initial.integer),ptr(b.keys),ptr(b.values),ptr(b.summaries),ptr(b.weights),
    ptr(s.mapping),ptr(s.integer),ptr(raw),ptr(error),b.batch,s.length,b.nodes,b.capacity,b.entries,fw),
    "explicit actual KV bank raw decoder");},{table,b.initial.floating,b.initial.integer,b.keys,b.values,b.summaries,b.weights,s.mapping,s.integer,raw,error});
  return {raw,s.integer};
}
BlockKVSequenceTape append_block_kv_bank_sequence(DeviceProgram& p,const weighted::KVConfig& config,
    const WeightedRawState& initial,const at::Tensor& content,const at::Tensor& times,const at::Tensor& owners,
    const at::Tensor& present,const WeightedKVParameters& weights,const at::Tensor& error,int64_t budget){
  config.validate();if(!present.defined()||present.dim()!=3||!initial.integer.defined()||initial.integer.dim()!=2||initial.integer.size(1)<=4)
    throw std::invalid_argument("KV bank requires B,L,N and bounded initial raw state");
  const auto b=present.size(0),l=present.size(1),n=present.size(2),c=initial.integer.size(1)-4;
  const auto maximum=std::numeric_limits<int64_t>::max();
  if(b<1||l<1||n<1||c>maximum-l||b>maximum/n||b*n>maximum/(c+l))throw std::invalid_argument("KV bank extent overflow");
  const auto o=b*n,rows=o*l,k=c+l,d=config.width,ds=config.summary_width;
  WeightedRawLayout layout(config,c),one(config,1);validate_weighted_raw_buffers(layout,initial);
  check(error,error,at::kInt,{1});check(present,error,at::kBool,{b,l,n});check(times,error,at::kLong,{b,l,n});
  check(owners,error,at::kLong,{b,l,n});check(content,error,at::kFloat,{rows,d+1});
  check(initial.floating,error,at::kFloat,{o,layout.floating_width()});check(initial.integer,error,at::kLong,{o,4+c});
  const WeightedStorageIncrement storage(p,{initial.floating,initial.integer,content,times,owners,present,
    weights.base,weights.left,weights.right,weights.summary,error});
  const long double reserve=o*k*(4.L*(2*d+ds+1)+9)+rows*(16.L*c+16.L*(4+c)+1)+
    o*(4.L*layout.floating_width()+8.L*c+8.L*(4+c)+8)+320+rows*(4.L*one.floating_width()+40);
  admit(reserve,budget);
  auto empty=at::zeros({rows,one.floating_width()},content.options()),empty_i=at::zeros({rows,5},times.options());empty_i.select(1,0).fill_(-1);
  auto write=append_weighted_kv_update(p,config,{empty,empty_i},content,times.reshape({rows}),owners.reshape({rows}),present.reshape({rows}),
    weights,error,budget-static_cast<int64_t>(reserve));
  auto bank=std::make_shared<BlockKVBank>(BlockKVBank{b,l,n,c,k,layout,initial});
  bank->keys=at::zeros({o,k,d},content.options());bank->values=at::zeros_like(bank->keys);bank->summaries=at::zeros({o,k,ds},content.options());
  bank->weights=at::zeros({o,k},content.options());bank->times=at::zeros({o,k},times.options());bank->valid=at::zeros({o,k},present.options());
  bank->writes=at::zeros({rows},present.options());bank->old_mapping=at::full({rows,c},-1,times.options());bank->proposal_mapping=at::full_like(bank->old_mapping,-1);
  bank->old_integer=at::zeros({rows,4+c},times.options());bank->proposal_integer=at::zeros_like(bank->old_integer);
  auto table=fields(layout,error);const auto actual=*bank;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_bank_plan)(1,stream,
    ptr(table),ptr(initial.floating),ptr(initial.integer),ptr(write.raw.result.floating),ptr(write.raw.result.integer),ptr(times),
    ptr(actual.keys),ptr(actual.values),ptr(actual.summaries),ptr(actual.weights),ptr(actual.times),ptr(actual.valid),ptr(actual.writes),
    ptr(actual.old_mapping),ptr(actual.proposal_mapping),ptr(actual.old_integer),ptr(actual.proposal_integer),ptr(error),
    b,l,n,c,d,ds,layout.floating_width(),one.floating_width(),config.window,0),"unique KV entry bank and actual raw snapshot indices");},
    {table,initial.floating,initial.integer,write.raw.result.floating,write.raw.result.integer,times,bank->keys,bank->values,bank->summaries,
      bank->weights,bank->times,bank->valid,bank->writes,bank->old_mapping,bank->proposal_mapping,bank->old_integer,bank->proposal_integer,error});
  std::vector<int64_t> last;for(int64_t sample=0;sample<b;++sample)for(int64_t node=0;node<n;++node)last.push_back((sample*l+l-1)*n+node);
  auto last_ids=at::tensor(last,at::kLong).to(error.device());
  auto last_map=at::zeros({o,c},times.options()),last_i=at::zeros({o,4+c},times.options());
  p.index_select(bank->proposal_mapping,0,last_ids,last_map);p.index_select(bank->proposal_integer,0,last_ids,last_i);
  bank->final_mapping=last_map;
  const auto used=storage.bytes();admit(used,budget);
  auto final=append_block_kv_bank_decode(p,{bank,last_map,last_i,1,l},error,budget-used);
  BlockKVSequenceTape out{config,b,l,n,layout,std::move(write),{{},bank->old_integer},{{},bank->proposal_integer},
    final,table,{},initial.integer,bank};
  admit(storage.bytes(),budget);return out;
}
BlockKVSnapshot append_block_kv_bound_bank(DeviceProgram& p,const weighted::KVConfig& config,const WeightedRawState& initial,
    int64_t b,int64_t l,int64_t n,const at::Tensor& error,int64_t budget){
  if(b<1||l<1||n<1||!initial.integer.defined()||initial.integer.dim()!=2||initial.integer.size(1)<=4)
    throw std::invalid_argument("KV bound bank dimensions");
  const auto c=initial.integer.size(1)-4,o=b*n,rows=o*l,d=config.width,ds=config.summary_width;WeightedRawLayout layout(config,c);
  check(initial.floating,error,at::kFloat,{o,layout.floating_width()});check(initial.integer,error,at::kLong,{o,4+c});
  const WeightedStorageIncrement storage(p,{initial.floating,initial.integer,error});
  admit(o*c*(4.L*(2*d+ds+1)+9)+rows*(32.L*c+73)+160,budget);
  auto bank=std::make_shared<BlockKVBank>(BlockKVBank{b,l,n,c,c,layout,initial});
  bank->keys=at::zeros({o,c,d},initial.floating.options());bank->values=at::zeros_like(bank->keys);bank->summaries=at::zeros({o,c,ds},initial.floating.options());
  bank->weights=at::zeros({o,c},initial.floating.options());bank->times=at::zeros({o,c},initial.integer.options());bank->valid=at::zeros({o,c},error.options().dtype(at::kBool));
  bank->writes=at::zeros({rows},bank->valid.options());bank->old_mapping=at::full({rows,c},-1,initial.integer.options());bank->proposal_mapping=at::full_like(bank->old_mapping,-1);
  bank->old_integer=at::zeros({rows,4+c},initial.integer.options());bank->proposal_integer=at::zeros_like(bank->old_integer);
  auto table=fields(layout,error),dummy=at::zeros({1},initial.floating.options()),dummy_i=at::zeros({5},initial.integer.options());const auto actual=*bank;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_kv_bank_plan)(1,stream,
    ptr(table),ptr(initial.floating),ptr(initial.integer),ptr(dummy),ptr(dummy_i),ptr(dummy_i),ptr(actual.keys),ptr(actual.values),ptr(actual.summaries),ptr(actual.weights),
    ptr(actual.times),ptr(actual.valid),ptr(actual.writes),ptr(actual.old_mapping),ptr(actual.proposal_mapping),ptr(actual.old_integer),ptr(actual.proposal_integer),ptr(error),
    b,l,n,c,d,ds,layout.floating_width(),0,config.window,1),"bound initial KV bank and repeated Read0 snapshot views");},
    {table,initial.floating,initial.integer,dummy,dummy_i,bank->keys,bank->values,bank->summaries,bank->weights,bank->times,bank->valid,bank->writes,
     bank->old_mapping,bank->proposal_mapping,bank->old_integer,bank->proposal_integer,error});
  admit(storage.bytes(),budget);return {bank,bank->old_mapping,bank->old_integer,l,0};
}
}
