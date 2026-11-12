#include "block_settle_flow.h"
#include "block_read_selection_internal.h"
#include "device_launch_tide_block_flow_fibers.h"
#include "device_launch_tide_block_flow_publish.h"
#include <algorithm>
#include <cmath>
#include <limits>
namespace tide::device_online {
namespace {
using namespace block_selection_detail;
struct Tables {std::vector<int64_t> nodes,atoms,counts,physical;int64_t capacity=1;};
Tables tables(const Graph& g,int64_t region,const BlockSettleKVBinding& binding){
  Tables out;for(size_t node=0;node<g.nodes.size();++node)if(g.nodes[node].region==region)out.nodes.push_back(node);
  if(out.nodes.empty()||binding.node_owners.size()!=out.nodes.size()||binding.input_owners.size()!=g.inputs.size()||binding.edge_owners.size()!=g.edges.size())
    throw std::invalid_argument("resident block concrete owner table differs from graph");
  std::vector<std::vector<std::array<int64_t,5>>> rows;
  for(auto node:out.nodes){std::vector<std::array<int64_t,5>> atoms;
    for(size_t port=0;port<g.inputs.size();++port)if(g.inputs[port]==node)
      atoms.push_back({g.source_domain->input[port],0,int64_t(port),binding.input_owners[port],int64_t(port)});
    for(size_t edge=0;edge<g.edges.size();++edge)if(g.edges[edge].target==node)
      atoms.push_back({g.source_domain->edge_target[edge],1,g.edges[edge].source,binding.edge_owners[edge],int64_t(edge)});
    std::sort(atoms.begin(),atoms.end());out.capacity=std::max<int64_t>(out.capacity,atoms.size());out.counts.push_back(atoms.size());rows.push_back(std::move(atoms));
  }
  for(const auto& row:rows){for(const auto& atom:row){if(atom[3]<0)throw std::invalid_argument("negative physical source owner");
      out.atoms.insert(out.atoms.end(),{atom[1],atom[2],atom[3]});out.physical.insert(out.physical.end(),{atom[1],atom[4]});}
    for(int64_t pad=row.size();pad<out.capacity;++pad){out.atoms.insert(out.atoms.end(),{0,0,0});out.physical.insert(out.physical.end(),{-1,-1});}}
  return out;
}
BlockPhysicalColumns allocate(int64_t frames,int64_t count,int64_t width,const at::Tensor& like){
  return {at::zeros({frames,count,width},like.options()),at::zeros({frames,count},like.options().dtype(at::kBool)),at::zeros({frames,count},like.options().dtype(at::kLong))};
}
void publish(DeviceProgram& p,int64_t command,const at::Tensor& nodes,const at::Tensor& strength,
    const at::Tensor& content,const at::Tensor& present,const at::Tensor& times,const BlockPhysicalColumns& out,
    const at::Tensor& error,int64_t frames,int64_t count,int64_t all_nodes,int64_t width){
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_flow_publish)(1,stream,
    ptr(nodes),ptr(strength),ptr(content),ptr(present),ptr(times),ptr(out.payload),ptr(out.present),ptr(out.times),ptr(error),
    command,frames,count,all_nodes,width),"physical block publication after guarded commit");},
    {nodes,strength,content,present,times,out.payload,out.present,out.times,error});
}
}
BlockSettleFlowTape append_block_settle_flow(DeviceProgram& p,TaskServices& services,const SettleGraph& spec,
    const std::vector<BlockSettleKVBinding>& bindings,const BlockSettleFlowInput& input,int64_t budget){
  using namespace block_selection_detail;const auto& g=spec.graph();const int64_t regions=g.regions.size(),nodes=g.nodes.size(),ports=g.inputs.size();
  if(spec.binding()!="ports"||!g.origins.empty()||!g.source_domain||regions<1||nodes<1||bindings.size()!=size_t(regions)||
    input.initial.size()!=size_t(regions)||input.bound_initial.size()!=size_t(regions)||!input.positions.defined()||input.positions.dim()!=2)
    throw std::invalid_argument("resident block graph needs concrete aligned Settle ports and all private initial bindings");
  for(const auto& node:g.nodes)
    if(node.clear||node.identity||node.next_state!="adopt-v1"||node.emission!="broadcast"||
       node.emit_period!=1||!node.emit_phases.empty()||node.state_clock!=StateClock{})
      throw std::invalid_argument("resident KV block requires adopt, broadcast and the original logical clock");
  for(const auto& region:g.regions)if(!region.observe_all)
    throw std::invalid_argument("resident KV block requires complete-region observation");
  const auto batch=input.positions.size(0),length=input.positions.size(1),d=bindings[0].memory.width;
  if(batch<1||length<1||batch>std::numeric_limits<int64_t>::max()/length)throw std::invalid_argument("resident block graph frame count overflow");
  const auto frames=batch*length,width=d+1;const auto device=input.payload.device();
  auto error=at::zeros({1},input.positions.options().dtype(at::kInt));check(error,error,at::kInt,{1});
  check(input.positions,error,at::kLong,{batch,length});check(input.payload,error,at::kFloat,{batch,length,ports,width});check(input.present,error,at::kBool,{batch,length,ports});
  BlockDispatchPlan plan;plan.groups=regions;plan.batch_capacity=1;plan.journal_capacity=regions;plan.tensor_bytes=budget;
  for(int64_t r=0;r<regions;++r){bindings[r].memory.validate();bindings[r].regional.validate();
    const auto alpha=bindings[r].aggregate_alpha;
    if(alpha<0||!std::isfinite(alpha)||!std::isfinite(float(alpha))||(alpha>0&&float(alpha)==0))
      throw std::invalid_argument("resident block Aggregate alpha is outside its declared FP32 coefficient domain");
    if(bindings[r].memory.width!=d)throw std::invalid_argument("connected block graph payload widths differ");
    plan.tasks.push_back({r,0,r,spec.ranks()[r],1});}
  for(const auto& e:g.edges){if(spec.ranks()[g.nodes[e.target].region]-spec.ranks()[g.nodes[e.source].region]!=e.delay)
      throw std::invalid_argument("resident block graph requires exact Settle rank-delay alignment");
    plan.dependencies.push_back({g.nodes[e.source].region,g.nodes[e.target].region,1});}
  BlockSettleFlowTape out;out.dispatcher=std::make_shared<BlockDispatcher>(plan,device);const auto q=out.dispatcher;error=q->error();
  std::vector<Tensor> borrowed{input.positions,input.payload,input.present};
  for(size_t r=0;r<bindings.size();++r){const auto& w=bindings[r].parameters;
    auto xs=block_selection_detail::borrowed({input.initial[r],{},{},{},{},{}},w.message.read,error);borrowed.insert(borrowed.end(),xs.begin(),xs.end());
    borrowed.insert(borrowed.end(),{input.bound_initial[r].floating,input.bound_initial[r].integer,w.source.a,w.source.b,w.source.embedding,
      w.update.base,w.update.left,w.update.right,w.update.summary,w.message.full.query,w.message.full.residual.out,w.message.full.residual.gate,w.message.full.residual.up,w.message.full.residual.down});}
  for(const auto& b:bindings)if(b.parameters.source_factors){const auto& f=*b.parameters.source_factors;borrowed.insert(borrowed.end(),{f.a_left,f.a_right,f.b_left,f.b_right});}
  const auto initial_bytes=p.retained_tensor_bytes(borrowed,services.programs());
  auto storage_bytes=[&](){return p.retained_tensor_bytes(borrowed,services.programs())-initial_bytes;};
  // Register the dispatcher once before subsequent admission. Counting its
  // explicit bytes as well as retained program storage would charge it twice.
  q->append_reset(p);auto remaining=[&](){admit(storage_bytes(),budget);return budget-storage_bytes();};
  admit(frames*(nodes+g.edges.size()+g.outputs.size())*(4.L*width+9),remaining());
  out.nodes=allocate(frames,nodes,width,input.payload);out.edges=allocate(frames,g.edges.size(),width,input.payload);out.outputs=allocate(frames,g.outputs.size(),width,input.payload);
  out.positions=at::zeros_like(input.positions);p.copy(out.positions,input.positions);
  p.zero(out.nodes.payload);p.zero(out.nodes.present);p.zero(out.nodes.times);
  const auto head=p.label(),finish=p.label();std::vector<size_t> branches{finish};for(int64_t r=0;r<regions;++r)branches.push_back(p.label());
  p.mark(head);q->append_choose(p);p.branch(q->branch(),branches);
  for(int64_t r=0;r<regions;++r){p.mark(branches[r+1]);const auto table=tables(g,r,bindings[r]);const int64_t count=table.nodes.size(),a=table.capacity;
    BlockSettleRegionTape stage;const auto longs=input.positions.options();stage.nodes=at::tensor(table.nodes,at::kLong).to(device);
    stage.atoms=at::tensor(table.atoms,at::kLong).reshape({count,a,3}).to(device);stage.atom_counts=at::tensor(table.counts,at::kLong).to(device);
    stage.physical_atoms=at::tensor(table.physical,at::kLong).reshape({count,a,2}).to(device);
    stage.source_mapping=at::full({frames*count,a,3},-1,longs);auto bank=at::tensor(bindings[r].node_owners,at::kLong).to(device);
    admit(frames*count*(a*(4.L*width+32)+24),remaining());
    BlockKVLocalInput local{at::zeros({batch,length,count,a,width},input.payload.options()),at::zeros({batch,length,count},longs),
      at::zeros({batch,length,count,a},longs),at::zeros({batch,length,count},longs),at::zeros({batch,length,count},longs),input.initial[r],input.bound_initial[r]};
    const auto positions=out.positions;const auto global=out.nodes;const auto rank=spec.ranks()[r],stride=spec.stride();
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_block_flow_fibers)(1,stream,
      ptr(stage.atoms),ptr(stage.atom_counts),ptr(bank),ptr(positions),ptr(input.payload),ptr(input.present),
      ptr(global.payload),ptr(global.present),ptr(local.payload),ptr(local.lengths),ptr(local.source_owners),ptr(local.node_owners),ptr(local.times),ptr(stage.source_mapping),ptr(error),
      batch,length,count,a,width,ports,nodes,rank,stride),"complete physical fibers from actual incoming block messages");},
      {stage.atoms,stage.atom_counts,stage.physical_atoms,bank,positions,input.payload,input.present,global.payload,global.present,
        local.payload,local.lengths,local.source_owners,local.node_owners,local.times,stage.source_mapping,error});
    auto local_plan=make_settle_block_plan(std::vector<int64_t>(count,0),{0},{},{},0,1,1,remaining());
    auto& numerical=services.begin();
    stage.local=std::make_shared<BlockKVLocalTape>(append_block_kv_local(numerical,local_plan,bindings[r].memory,bindings[r].regional,
      bindings[r].read_mode,bindings[r].aggregate_alpha,local,bindings[r].parameters,error,remaining()));
    services.end(numerical);
    const auto& message=stage.local->messages.messages;
    publish(p,0,stage.nodes,message.strength,message.content,message.present,message.times,
      {out.nodes.payload,q->emitted(),out.nodes.times},error,frames,count,nodes,width);
    q->append_commit(p);
    publish(p,1,stage.nodes,message.strength,message.content,message.present,message.times,out.nodes,error,frames,count,nodes,width);
    p.branch(q->branch(),{finish,head});out.regions.push_back(std::move(stage));
  }
  p.mark(finish);
  auto ports_out=[&](const std::vector<int64_t>& source,const std::vector<int64_t>& delay,const BlockPhysicalColumns& target){
    auto s=at::tensor(source,at::kLong).to(device),ds=at::tensor(delay,at::kLong).to(device);
    publish(p,2,s,ds,out.nodes.payload,out.nodes.present,out.nodes.times,target,error,frames,source.size(),nodes,width);};
  std::vector<int64_t> sources,delays;for(const auto& edge:g.edges){sources.push_back(edge.source);delays.push_back(edge.delay);}
  ports_out(sources,delays,out.edges);ports_out(g.outputs,std::vector<int64_t>(g.outputs.size(),0),out.outputs);
  admit(storage_bytes(),budget);return out;
}
}
