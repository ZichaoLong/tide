#include "block_kv_messages_saved.h"
#include "block_read_selection_internal.h"
#include "device_launch_tide_weighted_row_accumulate.h"

namespace tide::device_online {
BlockKVMessageGradient append_block_kv_messages_vjp(DeviceProgram& p,const BlockKVMessageTape& t,
    const BlockKVMessageRoots& roots,const at::Tensor& error,int64_t budget){
  using namespace block_selection_detail;
  if(!t.columns||!t.messages.content.defined()||t.messages.content.dim()!=3)
    throw std::invalid_argument("missing KV message block tape");
  const auto frames=t.selection.frames,nodes=t.selection.nodes,rows=frames*nodes,d=t.messages.content.size(2);
  auto borrowed=block_message_detail::saved(t);borrowed.push_back(error);
  for(const auto* r:{&roots.controls,&roots.full,&roots.message_strength,&roots.message_content,&roots.projection}){
    borrowed.push_back(r->value);borrowed.push_back(r->connected);}
  const WeightedStorageIncrement storage(p,std::move(borrowed));
  auto remaining=[&](){const auto bytes=storage.bytes();admit(bytes,budget);return budget-bytes;};
  std::vector<WeightedRowRoot> bound;
  auto append_root=[&](const WeightedRowAdjoint& root,size_t index,bool vector,bool active){
    if(root.value.defined()!=root.connected.defined())throw std::invalid_argument("block root needs both cotangent and connection mask");
    if(!root.value.defined())return;
    if(vector)check(root.value,error,at::kFloat,{frames,nodes,d});else check(root.value,error,at::kFloat,{frames,nodes});
    check(root.connected,error,at::kBool,{frames,nodes});auto connected=root.connected;
    if(active){connected=at::zeros_like(root.connected);p.logical_and(root.connected,t.messages.present,connected);}
    bound.push_back({index,root.value.reshape({rows,vector?d:1}),connected.reshape({rows})});
  };
  append_root(roots.controls,t.strength_index,false,false);
  append_root(roots.full,t.content_index,true,true);
  append_root(roots.message_strength,t.strength_index,false,true);
  append_root(roots.message_content,t.content_index,true,true);
  if(roots.projection.value.defined()||roots.projection.connected.defined()){
    if(!t.projection_index)throw std::invalid_argument("cG roots require the explicit numerical projection");
    append_root(roots.projection,*t.projection_index,true,true);
  }
  auto binding=t.columns->reverse(p,bound,error,remaining());
  const auto c=binding.rows.at(t.strength_index),g=binding.rows.at(t.content_index);
  auto read=append_block_read_selection_vjp(p,t.selection,c.value.reshape({frames,nodes}),c.connected.reshape({frames,nodes}),error,remaining());
  auto full=append_weighted_kv_full_vjp(p,t.full,g.value,g.connected,error,remaining());
  WeightedRowAdjoint content{at::zeros_like(read.content).reshape({rows,d}),at::zeros_like(read.content_connected).reshape({rows})};
  p.zero(content.value);p.zero(content.connected);const auto owners=t.selection.owners;
  for(const auto& source:std::vector<WeightedRowAdjoint>{{read.content.reshape({rows,d}),read.content_connected.reshape({rows})},
      {full.input,full.input_connected}}){
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_row_accumulate)(1,stream,
      ptr(owners),ptr(source.value),ptr(source.connected),ptr(content.value),ptr(content.connected),ptr(error),rows,d),
      "block Read/Full common content adjoint");},{owners,source.value,source.connected,content.value,content.connected,error});
  }
  content.value=content.value.reshape({frames,nodes,d});content.connected=content.connected.reshape({frames,nodes});
  admit(storage.bytes(),budget);return {std::move(read),std::move(full),std::move(content)};
}
}
