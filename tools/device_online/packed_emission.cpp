#include "packed_emission.h"
#include "content_profile.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_emission_plan.h"
#include "aclrtlaunch_tide_emission_chunk.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
std::pair<long double,long double> footprint(int64_t parameters,int64_t width,int64_t capacity,int64_t slots,int64_t nodes) {
  const long double persistent=parameters?4.L*(parameters+1.L)*(width*static_cast<long double>(width)+width):0;
  return {persistent+128.L*capacity+16.L*(capacity+1.L)*width+64.L*(slots+nodes+1.L),
    parameters?8.L*width*width+80.L*width+64:0.L};
}
}
long double PackedEmission::minimum_bytes(const ContentProfile& p,int64_t arrivals,int64_t outputs) {
  int64_t parameters=0;const auto& g=p.graph;
  for(size_t n=0;n<g.nodes.size();++n)if(!g.nodes[n].identity&&g.nodes[n].emission=="slot_affine")
    parameters+=g.outgoing_ports.offsets[n+1]-g.outgoing_ports.offsets[n];
  const auto [fixed,row]=footprint(parameters,p.width,arrivals+outputs,g.outgoing_ports.bindings.size(),g.nodes.size());
  return fixed+row;
}
PackedEmission::PackedEmission(const ContentProfile& profile,at::Device device,int64_t samples,
    int64_t arrivals,int64_t outputs,int64_t max_rows,int64_t budget)
    :nodes_(profile.graph.nodes.size()),samples_(samples),width_(profile.width),arrivals_(arrivals),outputs_(outputs),
     slots_(profile.graph.outgoing_ports.bindings.size()),parameters_(0),chunk_(0),reserved_(0) {
  const auto maximum=std::numeric_limits<int64_t>::max();
  if(at::GradMode::is_enabled()||device.type()!=c10::DeviceType::PrivateUse1||samples<1||arrivals<1||outputs<1||max_rows<1
      ||budget<1||arrivals>maximum-outputs||width_<1||nodes_<1)
    throw std::invalid_argument("packed emission requires bounded dimensions, no-grad NPU FP32");
  capacity_=arrivals+outputs;
  const auto& g=profile.graph;std::vector<int64_t> table,periods;std::vector<at::Tensor> weights,biases,scales;
  for(int64_t node=0;node<nodes_;++node) {
    const auto& n=g.nodes[node];const auto& w=profile.model.nodes[node];periods.push_back(n.emit_period);
    for(int64_t j=g.outgoing_ports.offsets[node];j<g.outgoing_ports.offsets[node+1];++j) {
      const int64_t slot=j-g.outgoing_ports.offsets[node];const auto binding=g.outgoing_ports.bindings[j];
      const int64_t phase=n.identity||n.emit_phases.empty()?-1:n.emit_phases[slot];
      int64_t parameter=-1;
      if(!n.identity&&n.emission=="slot_affine") {
        parameter=parameters_++;weights.push_back(w.extra.at("emit_w_"+std::to_string(slot)));
        biases.push_back(w.extra.at("emit_b_"+std::to_string(slot)));
      }
      const int64_t target=binding.kind?g.edges[binding.id].target:node;
      const int64_t delay=binding.kind?g.edges[binding.id].delay:0;
      table.insert(table.end(),{phase,binding.kind,binding.id,target,delay,parameter});
      scales.push_back(binding.kind?profile.model.edge_scale[binding.id]:profile.model.output_scale[binding.id]);
    }
  }
  // Stage storage is linear in declared message capacities. Matrices are
  // gathered only for present affine rows and bounded by a separate chunk.
  const auto [fixed,per_row]=footprint(parameters_,width_,capacity_,slots_,nodes_);
  if(fixed+(parameters_?per_row:0)>budget)throw std::invalid_argument("packed emission exceeds workspace budget");
  if(parameters_)chunk_=static_cast<int64_t>(std::min<long double>(max_rows,(budget-fixed)/per_row));
  reserved_=static_cast<int64_t>(fixed+chunk_*per_row);
  auto options=at::TensorOptions().dtype(at::kFloat);
  scales.push_back(at::zeros({},options));
  offsets_=at::tensor(g.outgoing_ports.offsets,at::kLong).to(device);
  periods_=at::tensor(periods,at::kLong).to(device);
  if(table.empty())table.resize(6);
  slots_table_=at::tensor(table,at::kLong).reshape({-1,6}).to(device);
  scales_=at::stack(scales).reshape({slots_+1,1}).to(device);
  chunks_=at::zeros({1},offsets_.options());
  if(parameters_) {
    weights.push_back(at::zeros({width_,width_},options));biases.push_back(at::zeros({width_},options));
    weights_=at::stack(weights).to(device);biases_=at::stack(biases).to(device);
  }
}
EmissionBatch PackedEmission::append_stage(CannProgram& p,const ActionBatch& actions,const at::Tensor& error) {
  const auto rows=actions.valid.numel(),width=width_,capacity=capacity_,chunk=chunk_,parameters=parameters_;
  if(rows<1||actions.coordinates.sizes()!=at::IntArrayRef{rows,4}||actions.values.sizes()!=at::IntArrayRef{rows,width}
      ||actions.coordinates.scalar_type()!=at::kLong||actions.values.scalar_type()!=at::kFloat||actions.valid.scalar_type()!=at::kBool
      ||error.sizes()!=at::IntArrayRef{1}||error.scalar_type()!=at::kInt)
    throw std::invalid_argument("invalid packed emission actions");
  for(const auto& x:{actions.coordinates,actions.values,actions.valid,error})
    if(x.device()!=offsets_.device()||!x.is_contiguous()||x.requires_grad())throw std::invalid_argument("invalid emission buffer ownership");
  auto longs=offsets_.options(),floats=actions.values.options(),booleans=actions.valid.options();
  auto atoms=[&](int64_t n){return AtomBatch{at::zeros({n,6},longs),at::zeros({n,width},floats),at::zeros({n},booleans)};};
  EmissionBatch out{at::zeros({capacity,6},longs),at::zeros({capacity+chunk+1,width},floats),at::zeros({1},longs),
    atoms(arrivals_),atoms(outputs_)};
  auto source=at::empty({capacity},longs),edge_source=at::empty({arrivals_},longs),edge_scale=at::empty_like(edge_source);
  auto output_source=at::empty({outputs_},longs),output_scale=at::empty_like(output_source),branch=at::zeros_like(error);
  auto values=at::zeros({rows+1,width},floats);
  const auto offsets=offsets_,periods=periods_,table=slots_table_;
  const auto nodes=nodes_,samples=samples_,slots=slots_,arrivals=arrivals_,outputs=outputs_;
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_emission_plan)(1,stream,
    ptr(actions.coordinates),ptr(actions.valid),ptr(offsets),ptr(periods),ptr(table),ptr(out.meta),ptr(out.count),ptr(source),
    ptr(out.arrivals.coordinates),ptr(out.arrivals.valid),ptr(edge_source),ptr(edge_scale),ptr(out.outputs.coordinates),
    ptr(out.outputs.valid),ptr(output_source),ptr(output_scale),ptr(branch),ptr(error),rows,nodes,samples,slots,capacity,arrivals,outputs),
    "plan present emission slots");},{actions.coordinates,actions.valid,offsets,periods,table,out.meta,out.count,source,
      out.arrivals.coordinates,out.arrivals.valid,edge_source,edge_scale,out.outputs.coordinates,out.outputs.valid,output_source,output_scale,branch,error});
  auto emit=p.label(),done=p.label();p.branch(branch,{done,emit});p.mark(emit);
  p.copy(values.narrow(0,0,rows),actions.values);p.index_select(values,0,source,out.values.narrow(0,0,capacity));
  if(parameters) {
    auto cursor=at::zeros({1},longs),zero=at::zeros_like(cursor),go=at::zeros_like(error);
    auto source_rows=at::empty({chunk},longs),parameter_rows=at::empty_like(source_rows),destinations=at::empty_like(source_rows);
    auto x=at::empty({chunk,width},floats),bias=at::empty_like(x),result=at::empty_like(x);
    auto weights=at::empty({chunk,width,width},floats);const auto batches=chunks_;
    p.copy(cursor,zero);auto head=p.label(),body=p.label(),end=p.label();p.mark(head);
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_emission_chunk)(1,stream,
      ptr(out.meta),ptr(out.count),ptr(cursor),ptr(source_rows),ptr(parameter_rows),ptr(destinations),ptr(go),ptr(batches),ptr(error),
      capacity,parameters,chunk),"pack selected emission projections");},
      {out.meta,out.count,cursor,source_rows,parameter_rows,destinations,go,batches,error});
    p.branch(go,{end,body});p.mark(body);
    p.index_select(out.values,0,source_rows,x);p.index_select(weights_,0,parameter_rows,weights);p.index_select(biases_,0,parameter_rows,bias);
    p.batch_matmul(x.reshape({chunk,1,width}),weights,result.reshape({chunk,1,width}));p.add(result,bias);
    p.index_copy(out.values,0,destinations,result);p.branch(go,{head});p.mark(end);
  }
  // Physical scales are applied after projection. Preserve the unscaled slot
  // values even for scale=0; diagnostics cannot reconstruct them by division.
  auto deliver=[&](const AtomBatch& target,const at::Tensor& order,const at::Tensor& scale_order) {
    auto selected=at::empty_like(target.values),scales=at::empty({target.values.size(0),1},floats);
    p.index_select(out.values,0,order,selected);p.index_select(scales_,0,scale_order,scales);p.multiply(selected,scales,target.values);
  };
  deliver(out.arrivals,edge_source,edge_scale);deliver(out.outputs,output_source,output_scale);p.mark(done);
  out.values=out.values.narrow(0,0,capacity);return out;
}
} // namespace tide::device_online
