#include "block_kv_local.h"
#include "block_read_selection_internal.h"
#include <limits>
namespace tide::device_online {
BlockKVLocalTape append_block_kv_local(DeviceProgram& p,const SettleBlockPlan& plan,
    const weighted::KVConfig& kv,const weighted::RegionalConfig& region,const std::string& mode,
    float alpha,const BlockKVLocalInput& input,const BlockKVLocalParameters& parameters,
    const at::Tensor& error,int64_t budget,bool projection){
  using namespace block_selection_detail;
  if(mode!="0"&&mode!="-"&&mode!="+")throw std::invalid_argument("KV local block requires explicit Read0/Read-/Read+");
  if(!input.payload.defined()||input.payload.dim()!=5||input.payload.size(4)!=kv.width+1)
    throw std::invalid_argument("KV local block requires B,L,N,atom,payload layout");
  const auto b=input.payload.size(0),l=input.payload.size(1),n=input.payload.size(2),a=input.payload.size(3),d=kv.width;
  if(b<1||l<1||n<1||a<1||b>std::numeric_limits<int64_t>::max()/n||b*n>std::numeric_limits<int64_t>::max()/l)
    throw std::invalid_argument("invalid KV local block dimensions");
  const auto rows=b*l*n;
  check(input.payload,error,at::kFloat,{b,l,n,a,d+1});check(input.lengths,error,at::kLong,{b,l,n});
  check(input.source_owners,error,at::kLong,{b,l,n,a});check(input.times,error,at::kLong,{b,l,n});
  check(input.node_owners,error,at::kLong,{b,l,n});
  auto borrowed=block_selection_detail::borrowed({input.initial,at::Tensor(),at::Tensor(),at::Tensor(),input.times,input.node_owners},parameters.message.read,error);
  borrowed.insert(borrowed.end(),{input.payload,input.lengths,input.source_owners,input.bound_initial.floating,input.bound_initial.integer,
    parameters.update.base,parameters.update.left,parameters.update.right,parameters.update.summary,
    parameters.source.a,parameters.source.b,parameters.source.embedding,parameters.message.full.query,
    parameters.message.full.residual.out,parameters.message.full.residual.gate,parameters.message.full.residual.up,parameters.message.full.residual.down});
  if(parameters.source_factors){const auto& f=*parameters.source_factors;borrowed.insert(borrowed.end(),{f.a_left,f.a_right,f.b_left,f.b_right});}
  const WeightedStorageIncrement storage(p,std::move(borrowed));auto remaining=[&](){admit(storage.bytes(),budget);return budget-storage.bytes();};
  auto source=parameters.source;std::optional<WeightedSourceAdaptation> adaptation;
  if(alpha>0&&parameters.source_factors){adaptation=append_weighted_source_adaptation(p,source,*parameters.source_factors,error,remaining());source=adaptation->effective;}
  auto aggregate=append_weighted_aggregate(p,input.payload.reshape({rows,a,d+1}),input.lengths.reshape({rows}),
    input.source_owners.reshape({rows,a}),source,alpha,kv.epsilon,error,remaining());
  auto present=at::zeros({b,l,n},input.lengths.options().dtype(at::kBool));
  p.less(at::zeros_like(input.lengths),input.lengths,present);
  auto sequence=append_block_kv_sequence(p,kv,input.initial,aggregate.output,input.times,input.node_owners,present,parameters.update,error,remaining());
  WeightedRawState read=mode=="+"?sequence.proposal:sequence.old;auto indices=at::zeros({rows},input.times.options());
  if(mode=="0"){
    validate_weighted_raw_buffers(sequence.layout,input.bound_initial);
    check(input.bound_initial.floating,error,at::kFloat,{b*n,sequence.layout.floating_width()});
    check(input.bound_initial.integer,error,at::kLong,{b*n,sequence.layout.integer_width()});
    std::vector<int64_t> ids;for(int64_t sample=0;sample<b;++sample)for(int64_t step=0;step<l;++step)
      for(int64_t node=0;node<n;++node)ids.push_back(sample*n+node);
    indices=at::tensor(ids,at::kLong).to(error.device());
    read={at::zeros_like(sequence.old.floating),at::zeros_like(sequence.old.integer)};
    p.index_select(input.bound_initial.floating,0,indices,read.floating);p.index_select(input.bound_initial.integer,0,indices,read.integer);
  }
  auto mass=columns(p,aggregate.output,0,1).reshape({b*l,n}),content=columns(p,aggregate.output,1,d).reshape({b*l,n,d});
  BlockKVMessageInput x{{read,mass,content,present.reshape({b*l,n}),input.times.reshape({b*l,n}),input.node_owners.reshape({b*l,n})},sequence.proposal,{}};
  auto messages=append_block_kv_messages(p,plan,kv,region,x,parameters.message,error,remaining(),projection);
  auto received=std::make_shared<WeightedRowGraph>(p,messages.selection.owners,parameters.update.base.size(0),error,remaining(),std::vector<at::Tensor>{aggregate.output});
  const auto h=received->input(aggregate.output),r=received->slice(h,0,1),u=received->slice(h,1,d);
  admit(storage.bytes(),budget);
  return {mode,std::move(aggregate),std::move(sequence),std::move(messages),read,indices,present,received,h,r,u,std::move(adaptation)};
}
}
