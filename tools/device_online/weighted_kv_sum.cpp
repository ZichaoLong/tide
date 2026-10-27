#include "weighted_kv_sum.h"
#include "device_launch_tide_weighted_sum_roots.h"
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}
void check(const at::Tensor& t,at::Device device,at::ScalarType type,at::IntArrayRef shape){
  if(!t.defined()||t.device()!=device||t.scalar_type()!=type||t.sizes()!=shape||!t.is_contiguous()||t.requires_grad())
    throw std::invalid_argument("invalid KV sum Mem buffer");
}
void budget(long double bytes,int64_t available){if(available<1||bytes>available)throw std::invalid_argument("KV sum Mem tensor budget exceeded");}
}
WeightedKVSumTape append_weighted_kv_sum(DeviceProgram& p,const weighted::KVConfig& c,const WeightedRawState& raw,
    const at::Tensor& time,const at::Tensor& present,const at::Tensor& error,int64_t available){
  c.validate();if(c.memory_read!="sum"||!raw.floating.defined()||raw.floating.dim()!=2||!raw.integer.defined()||raw.integer.dim()!=2||raw.integer.size(1)<=4)
    throw std::invalid_argument("KV sum Mem requires sum configuration and typed raw cache");
  const auto rows=raw.floating.size(0),capacity=raw.integer.size(1)-4;const auto device=raw.floating.device();
  budget(rows*(128.L*capacity*(static_cast<long double>(c.width)+c.summary_width+1)+256)
    +(raw.floating.scalar_type()==at::kHalf?4.L*raw.floating.numel():0),available);
  auto window=append_weighted_kv_window(p,c,raw,time,present,error,available);
  std::vector<int64_t> ids;for(int64_t r=0;r<rows;++r){ids.push_back(r);ids.push_back(r);}
  auto indices=at::tensor(ids,at::kLong).to(device),values=at::zeros({2*rows,capacity,c.summary_width},window.summaries.options());
  auto lengths=at::zeros({2*rows},raw.integer.options()),selected=at::zeros({2*rows},present.options());
  p.index_select(window.summaries,0,indices,values);p.index_select(window.lengths,0,indices,lengths);p.index_select(window.present,0,indices,selected);
  auto pool=append_weighted_pool(p,window.weights.reshape({2*rows,capacity}),values,{},lengths,selected,error,available);
  return {window,pool,pool.output.reshape({rows,2,c.summary_width}),pool.total.reshape({rows,2})};
}
WeightedKVWindowGradient append_weighted_kv_sum_vjp(DeviceProgram& p,const WeightedKVSumTape& t,const at::Tensor& memory_bar,
    const at::Tensor& total_bar,const at::Tensor& roots,const at::Tensor& error,int64_t available){
  if(!t.memory.defined()||t.memory.dim()!=3||t.memory.size(1)!=2||t.memory.size(0)<1)
    throw std::invalid_argument("invalid KV sum tape");
  const auto device=t.memory.device();const auto rows=t.memory.size(0),ds=t.memory.size(2),capacity=t.window.layout.capacity(),d=t.window.layout.field("keys").width;
  check(memory_bar,device,at::kFloat,{rows,2,ds});check(total_bar,device,at::kFloat,{rows,2});check(roots,device,at::kBool,{rows,4});check(error,device,at::kInt,{1});
  budget(rows*(128.L*capacity*(static_cast<long double>(d)+ds+1)+256),available);
  auto indices=at::tensor(std::vector<int64_t>{0,2,1,3},at::kLong).to(device),pool_roots=at::zeros_like(roots);p.index_select(roots,1,indices,pool_roots);
  auto pool=append_weighted_pool_vjp(p,t.pool,memory_bar.reshape({2*rows,ds}),total_bar.reshape({2*rows}),pool_roots.reshape({2*rows,2}),error,available);
  auto summary=at::zeros({rows,capacity,ds},memory_bar.options()),window_roots=at::zeros({rows,5},roots.options());
  p.sum(pool.values.reshape({rows,2,capacity,ds}),1,false,summary);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_sum_roots)(1,stream,
    ptr(pool.connected),ptr(window_roots),ptr(error),rows),"KV sum independent short/long descriptor roots");},{pool.connected,window_roots,error});
  return append_weighted_kv_window_vjp(p,t.window,at::zeros_like(t.window.keys),at::zeros_like(t.window.values),summary,
    pool.weights.reshape({rows,2,capacity}),window_roots,error,available);
}
}
