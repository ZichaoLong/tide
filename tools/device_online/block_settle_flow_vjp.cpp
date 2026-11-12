#include "block_settle_flow_vjp.h"
#include "block_kv_local_saved.h"
#include "block_read_selection_internal.h"
#include "device_launch_tide_block_flow_reverse_ports.h"
#include "device_launch_tide_block_flow_reverse_gather.h"
#include "device_launch_tide_block_flow_reverse_fibers.h"
#include "device_launch_tide_block_flow_reverse_choose.h"
namespace tide::device_online {
BlockSettleFlowGradient append_block_settle_flow_vjp(DeviceProgram& p,TaskServices& services,const SettleGraph& spec,
    const BlockSettleFlowTape& t,const BlockSettleFlowRoots& roots,int64_t budget){
  using namespace block_selection_detail;const auto& graph=spec.graph();
  const int64_t frames=t.positions.numel(),nodes=graph.nodes.size(),regions=graph.regions.size(),width=t.nodes.payload.size(2),ports=graph.inputs.size();
  if(!t.dispatcher||t.regions.size()!=size_t(regions)||(!roots.regions.empty()&&roots.regions.size()!=size_t(regions)))
    throw std::invalid_argument("resident block reverse requires complete matching graph tape");
  const auto q=t.dispatcher;const auto error=q->error();
  std::vector<Tensor> borrowed{t.nodes.payload,t.nodes.present,t.nodes.times,t.edges.payload,t.edges.present,t.edges.times,
    t.outputs.payload,t.outputs.present,t.outputs.times,t.positions,q->journal(),q->ledger(),error};
  for(const auto& stage:t.regions){auto saved=block_local_detail::saved(*stage.local);borrowed.insert(borrowed.end(),saved.begin(),saved.end());
    borrowed.insert(borrowed.end(),{stage.nodes,stage.atoms,stage.atom_counts,stage.physical_atoms,stage.source_mapping});}
  auto add_roots=[&](const BlockPhysicalRoots& r){for(const auto* x:{&r.strength,&r.content}){borrowed.push_back(x->value);borrowed.push_back(x->connected);}};
  add_roots(roots.edges);add_roots(roots.outputs);
  for(const auto& r:roots.regions){for(const auto* x:{&r.messages.controls,&r.messages.full,&r.messages.message_strength,&r.messages.message_content,&r.messages.projection,&r.received}){
      borrowed.push_back(x->value);borrowed.push_back(x->connected);}
    for(const auto* x:{&r.messages.read.memory,&r.messages.read.total}){borrowed.push_back(x->value);borrowed.push_back(x->connected);}
    for(const auto& [name,x]:r.messages.read.describe){borrowed.push_back(x.value);borrowed.push_back(x.connected);}
    for(const auto& x:{r.raw.old,r.raw.old_connected,r.raw.proposal,r.raw.proposal_connected,r.raw.final,r.raw.final_connected})borrowed.push_back(x);}
  const auto initial_bytes=p.retained_tensor_bytes(borrowed,services.programs());
  auto storage_bytes=[&](){return p.retained_tensor_bytes(borrowed,services.programs())-initial_bytes;};
  auto remaining=[&](){admit(storage_bytes(),budget);return budget-storage_bytes();};
  admit(frames*(nodes*(4.L*width+2)+ports*(4.L*width+1))+20,remaining());
  auto message=at::zeros_like(t.nodes.payload),message_roots=at::zeros({frames,nodes,2},t.nodes.present.options());
  BlockSettleFlowGradient out{at::zeros({frames,ports,width},message.options()),at::zeros({frames,ports},t.nodes.present.options()),{}};
  p.zero(message);p.zero(message_roots);p.zero(out.inputs);p.zero(out.input_connected);
  auto field=[&](const WeightedRowAdjoint& root,int64_t count,bool vector){
    if(root.value.defined()!=root.connected.defined())throw std::invalid_argument("physical block root requires value and mask");
    if(!root.value.defined())return WeightedRowAdjoint{at::zeros(vector?std::vector<int64_t>{frames,count,width-1}:std::vector<int64_t>{frames,count},message.options()),at::zeros({frames,count},message_roots.options())};
    if(vector)check(root.value,error,at::kFloat,{frames,count,width-1});else check(root.value,error,at::kFloat,{frames,count});
    check(root.connected,error,at::kBool,{frames,count});return root;
  };
  auto ports_roots=[&](const std::vector<int64_t>& sources,const Tensor& present,const BlockPhysicalRoots& original){
    const auto count=int64_t(sources.size());auto src=at::tensor(sources,at::kLong).to(error.device());
    const auto mass=field(original.strength,count,false),content=field(original.content,count,true);
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_flow_reverse_ports)(1,stream,
      ptr(src),ptr(present),ptr(mass.value),ptr(mass.connected),ptr(content.value),ptr(content.connected),ptr(message),ptr(message_roots),ptr(error),
      frames,count,nodes,width),"independent physical edge/port roots to source message");},
      {src,present,mass.value,mass.connected,content.value,content.connected,message,message_roots,error});
  };
  std::vector<int64_t> sources;for(const auto& e:graph.edges)sources.push_back(e.source);
  ports_roots(sources,t.edges.present,roots.edges);ports_roots(graph.outputs,t.outputs.present,roots.outputs);
  auto cursor=at::zeros({1},q->ledger().options()),branch=at::zeros({1},error.options()),zero=at::zeros_like(branch);
  auto choose=[&](int64_t command){const auto journal=q->journal(),ledger=q->ledger();
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_flow_reverse_choose)(1,stream,
      ptr(journal),ptr(ledger),ptr(cursor),ptr(branch),ptr(error),regions,command),"reverse actual resident block journal");},
      {journal,ledger,cursor,branch,error});};
  choose(0);const auto head=p.label(),finish=p.label();std::vector<size_t> labels{finish};for(int64_t r=0;r<regions;++r)labels.push_back(p.label());
  p.mark(head);choose(1);p.branch(branch,labels);
  for(int64_t r=0;r<regions;++r){p.mark(labels[r+1]);const auto& stage=t.regions[r];const auto& local=*stage.local;
    const auto count=local.sequence.nodes,capacity=local.aggregate.payload.size(1),rows=frames*count;
    auto bound=roots.regions.empty()?BlockKVLocalRoots{}:roots.regions[r];
    const auto direct_mass=field(bound.messages.message_strength,count,false),direct_content=field(bound.messages.message_content,count,true);
    bound.messages.message_strength={at::zeros({frames,count},message.options()),at::zeros({frames,count},message_roots.options())};
    bound.messages.message_content={at::zeros({frames,count,width-1},message.options()),at::zeros({frames,count},message_roots.options())};
    const auto ids=stage.nodes,active=local.messages.messages.present;const auto mb=bound.messages.message_strength,cb=bound.messages.message_content;
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_flow_reverse_gather)(1,stream,
      ptr(ids),ptr(active),ptr(message),ptr(message_roots),ptr(direct_mass.value),ptr(direct_mass.connected),ptr(direct_content.value),ptr(direct_content.connected),
      ptr(mb.value),ptr(mb.connected),ptr(cb.value),ptr(cb.connected),ptr(error),frames,count,nodes,width),"downstream and independent source-message roots");},
      {ids,active,message,message_roots,direct_mass.value,direct_mass.connected,direct_content.value,direct_content.connected,mb.value,mb.connected,cb.value,cb.connected,error});
    auto& numerical=services.begin();
    auto gradient=append_block_kv_local_vjp(numerical,local,bound,error,remaining());
    services.end(numerical);
    const auto mapping=stage.source_mapping,lengths=local.aggregate.lengths,values=gradient.aggregate.payload,flags=gradient.aggregate.connected;
    const auto inputs=out.inputs,input_roots=out.input_connected;
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_flow_reverse_fibers)(1,stream,
      ptr(mapping),ptr(lengths),ptr(values),ptr(flags),ptr(inputs),ptr(input_roots),ptr(message),ptr(message_roots),ptr(error),
      rows,capacity,width,frames*ports,frames*nodes),"transpose actual physical atom routing with duplicate-edge accumulation");},
      {mapping,lengths,values,flags,inputs,input_roots,message,message_roots,error});
    p.branch(zero,{head});out.regions.push_back(std::move(gradient));
  }
  p.mark(finish);admit(storage_bytes(),budget);return out;
}
}
