#include "reverse_links.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_reverse_links.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}
void tensor(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape) {
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid reverse-link buffer");
}
}
ReverseLinks append_reverse_links(CannProgram& p,const ReverseTape& t,const at::Tensor& error,int64_t budget) {
  if(at::GradMode::is_enabled()||!t.graph||!t.state.metadata.defined()||t.state.metadata.dim()!=2
      ||!t.fiber_meta.defined()||t.fiber_meta.dim()!=2||!t.pending.valid.defined()||!t.outputs.valid.defined())
    throw std::invalid_argument("reverse links require actual no-grad device journals");
  const auto device=t.state.metadata.device();const auto& g=*t.graph;
  const auto payload=t.source_scales.scalar_type();
  if(payload!=at::kFloat&&payload!=at::kHalf)throw std::invalid_argument("reverse links require FP32/FP16 forward payloads");
  const int64_t capacity=t.state.metadata.size(0),fibers=t.fiber_meta.size(0),pending=t.pending.valid.numel(),outputs=t.outputs.valid.numel();
  const int64_t nodes=g.nodes.size(),inputs=g.inputs.size(),edges=g.edges.size(),ports=g.outputs.size(),samples=t.state.samples;
  const int64_t parameters=inputs+2*edges+ports;
  const long double total=fibers+static_cast<long double>(pending)+outputs;
  const long double bytes=96.L*capacity+80.L*total+16.L*(parameters+1)+8.L*samples*nodes+64.L*(edges+ports+1);
  if(device.type()!=c10::DeviceType::PrivateUse1||budget<1||bytes>budget||capacity<1||fibers<1||pending<1||outputs<1
      ||nodes<1||samples<1||t.cut<0||t.stop<t.cut)
    throw std::invalid_argument("reverse-link tensor budget exceeded or invalid dimensions");
  const int64_t messages=fibers+pending+outputs,width=t.full.width;
  tensor(t.state.metadata,device,at::kLong,{capacity,13});tensor(t.state.count,device,at::kLong,{1});
  tensor(t.fiber_meta,device,at::kLong,{fibers,6});tensor(t.fiber_count,device,at::kLong,{1});
  tensor(t.sources,device,at::kLong,{std::max<int64_t>(1,inputs+edges),2});
  tensor(t.source_scales,device,payload,{std::max<int64_t>(1,inputs+edges)});
  tensor(t.delivery_scales,device,payload,{edges+ports+1,1});
  for(const auto& a:{t.pending,t.outputs}){tensor(a.coordinates,device,at::kLong,{a.valid.numel(),6});tensor(a.valid,device,at::kBool,{a.valid.numel()});
    tensor(a.values,device,payload,{a.valid.numel(),width});}
  tensor(t.pending_count,device,at::kLong,{1});tensor(t.output_count,device,at::kLong,{1});tensor(error,device,at::kInt,{1});
  std::vector<int64_t> edge_table(std::max<int64_t>(1,edges)*4),port_table(std::max<int64_t>(1,ports)*2);
  for(int64_t n=0;n<nodes;++n)for(int64_t j=g.outgoing_ports.offsets[n];j<g.outgoing_ports.offsets[n+1];++j) {
    const auto binding=g.outgoing_ports.bindings[j];const auto id=binding.id;
    if(binding.kind){const auto& e=g.edges[id];edge_table[id*4]=e.source;edge_table[id*4+1]=e.target;edge_table[id*4+2]=e.delay;edge_table[id*4+3]=j;}
    else {port_table[id*2]=n;port_table[id*2+1]=j;}
  }
  auto longs=t.state.metadata.options(),booleans=t.pending.valid.options();
  auto edge_data=at::tensor(edge_table,at::kLong).reshape({-1,4}).to(device);
  auto port_data=at::tensor(port_table,at::kLong).reshape({-1,2}).to(device);
  int64_t buckets=1;while(buckets<2*capacity)buckets*=2;
  auto hash=at::empty({buckets},longs);
  ReverseLinks out{at::empty({messages,4},longs),at::empty({messages},booleans),at::empty({capacity},longs),
    at::empty({messages},longs),at::empty({capacity},longs),at::empty({fibers},longs),at::empty({std::max<int64_t>(1,parameters)},longs),
    at::empty({2*messages},longs),at::empty({capacity+1},longs),at::empty({1},longs),
    at::empty({std::max<int64_t>(1,parameters)},t.source_scales.options().dtype(at::kFloat)),fibers,pending,outputs,parameters};
  p.zero(out.scales);
  auto copy_scale=[&](const at::Tensor& dst,const at::Tensor& src){if(payload==at::kHalf)p.cast(src,dst);else p.copy(dst,src);};
  if(inputs+edges)copy_scale(out.scales.narrow(0,0,inputs+edges),t.source_scales);
  if(edges+ports)copy_scale(out.scales.narrow(0,inputs+edges,edges+ports),t.delivery_scales.narrow(0,0,edges+ports).view({-1}));
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_reverse_links)(1,stream,
    ptr(t.state.metadata),ptr(t.state.count),ptr(t.fiber_meta),ptr(t.fiber_count),ptr(t.pending.coordinates),ptr(t.pending.valid),ptr(t.pending_count),
    ptr(t.outputs.coordinates),ptr(t.outputs.valid),ptr(t.output_count),ptr(t.sources),ptr(edge_data),ptr(port_data),ptr(hash),
    ptr(out.messages),ptr(out.valid),ptr(out.producer_head),ptr(out.producer_next),ptr(out.consumer_head),ptr(out.consumer_next),
    ptr(out.scale_head),ptr(out.scale_next),ptr(out.stage_offsets),ptr(out.stages),ptr(error),
    capacity,fibers,pending,outputs,nodes,samples,inputs,edges,ports,buckets,t.cut,t.stop),"link actual reverse event/message dependencies");},
    {t.state.metadata,t.state.count,t.fiber_meta,t.fiber_count,t.pending.coordinates,t.pending.valid,t.pending_count,
     t.outputs.coordinates,t.outputs.valid,t.output_count,t.sources,edge_data,port_data,hash,out.messages,out.valid,
     out.producer_head,out.producer_next,out.consumer_head,out.consumer_next,out.scale_head,out.scale_next,out.stage_offsets,out.stages,error});
  return out;
}
} // namespace tide::device_online
