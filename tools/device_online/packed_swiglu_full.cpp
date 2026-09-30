#include "packed_swiglu_full.h"
#include "content_profile.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_full_plan.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
std::pair<long double,long double> footprint(int64_t parameters,int64_t width,int64_t rows,int64_t nodes) {
  return {24.L*(parameters+1.L)*width*width+16.L*(rows+1.L)*width+16.L*(nodes+1.L),24.L*width*width+128.L*width+128};
}
}
long double PackedSwiGluFull::minimum_bytes(const ContentProfile& p,int64_t capacity) {
  int64_t count=0;for(const auto& n:p.graph.nodes)count+=!n.identity&&n.full=="swiglu";
  if(!count)return 0;
  const auto [fixed,row]=footprint(count,p.width,capacity,p.graph.nodes.size());return fixed+row;
}
PackedSwiGluFull::PackedSwiGluFull(const ContentProfile& profile,at::Device device,int64_t capacity,int64_t max_rows,int64_t budget)
    :nodes_(profile.graph.nodes.size()),width_(profile.width),rows_(capacity),parameters_(0),chunk_(0),reserved_(0) {
  if(at::GradMode::is_enabled()||device.type()!=c10::DeviceType::PrivateUse1||capacity<1||max_rows<1||budget<1)
    throw std::invalid_argument("packed SwiGLU requires bounded dimensions and no-grad NPU");
  std::vector<int64_t> kinds,mapping;std::vector<at::Tensor> gate,up,down;
  for(int64_t n=0;n<nodes_;++n) {
    const auto& node=profile.graph.nodes[n];const bool enabled=!node.identity&&node.full=="swiglu";
    kinds.push_back(enabled);mapping.push_back(enabled?parameters_++:-1);
    if(enabled) {
      const auto& w=profile.model.nodes[n];gate.push_back(w.extra.at("ffn_gate"));
      up.push_back(w.extra.at("ffn_up"));down.push_back(w.extra.at("ffn_down"));
    }
  }
  if(!parameters_)throw std::invalid_argument("packed SwiGLU requires at least one declared owner");
  const auto [fixed,per_row]=footprint(parameters_,width_,rows_,nodes_);
  if(fixed+per_row>budget)throw std::invalid_argument("one packed SwiGLU row exceeds workspace budget");
  chunk_=static_cast<int64_t>(std::min<long double>({static_cast<long double>(max_rows),static_cast<long double>(capacity),(budget-fixed)/per_row}));
  reserved_=static_cast<int64_t>(fixed+per_row*chunk_);
  for(auto& index:mapping)if(index<0)index=parameters_;
  mapping.push_back(parameters_); // Planner's independent zero parameter sentinel.
  kinds_=at::tensor(kinds,at::kLong).to(device);mapping_=at::tensor(mapping,at::kLong).to(device);
  auto options=at::TensorOptions().dtype(at::kFloat);
  gate.push_back(at::zeros({width_,2*width_},options));up.push_back(at::zeros({width_,2*width_},options));
  down.push_back(at::zeros({2*width_,width_},options));
  gate_=at::stack(gate).to(device);up_=at::stack(up).to(device);down_=at::stack(down).to(device);
}
ActionBatch PackedSwiGluFull::append_stage(CannProgram& p,const ActionBatch& input,const at::Tensor& content,
    const at::Tensor& comparison,const at::Tensor& error,const at::Tensor& chunks) {
  const auto rows=rows_,width=width_,chunk=chunk_,nodes=nodes_;
  if(input.values.sizes()!=at::IntArrayRef{rows,width}||content.sizes()!=input.values.sizes()||comparison.sizes()!=input.values.sizes()
      ||input.values.scalar_type()!=at::kFloat||content.scalar_type()!=at::kFloat||comparison.scalar_type()!=at::kFloat
      ||input.coordinates.sizes()!=at::IntArrayRef{rows,4}||input.coordinates.scalar_type()!=at::kLong
      ||input.valid.sizes()!=at::IntArrayRef{rows}||input.valid.scalar_type()!=at::kBool
      ||error.sizes()!=at::IntArrayRef{1}||error.scalar_type()!=at::kInt||chunks.sizes()!=at::IntArrayRef{1}||chunks.scalar_type()!=at::kLong)
    throw std::invalid_argument("invalid SwiGLU buffers");
  for(const auto& x:{input.values,content,comparison,input.coordinates,input.valid,error,chunks})
    if(x.device()!=kinds_.device()||!x.is_contiguous()||x.requires_grad())throw std::invalid_argument("invalid SwiGLU buffer ownership");
  auto source=at::empty({chunk},kinds_.options()),owners=at::empty_like(source),parameters=at::empty_like(source),destination=at::empty_like(source);
  auto cursor=at::zeros({1},kinds_.options()),zero=at::zeros_like(cursor),branch=at::zeros_like(error);
  auto comparisons=at::zeros({rows+1,width},input.values.options()),contents=at::zeros_like(comparisons);
  auto output=at::zeros({rows+chunk,width},input.values.options());
  auto x=at::empty({chunk,1,width},input.values.options()),h=at::empty_like(x),result=at::empty_like(x);
  auto gate=at::empty({chunk,width,2*width},input.values.options()),up=at::empty_like(gate);
  auto down=at::empty({chunk,2*width,width},input.values.options());
  auto a=at::empty({chunk,1,2*width},input.values.options()),b=at::empty_like(a),activated=at::empty_like(a),product=at::empty_like(a);
  p.copy(cursor,zero);p.copy(comparisons.narrow(0,0,rows),comparison);p.copy(contents.narrow(0,0,rows),content);
  p.copy(output.narrow(0,0,rows),input.values);
  const auto kinds=kinds_;auto head=p.label(),body=p.label(),done=p.label();p.mark(head);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_full_plan)(1,stream,
    ptr(input.coordinates),ptr(input.valid),ptr(kinds),ptr(cursor),ptr(source),ptr(owners),ptr(destination),
    ptr(branch),ptr(chunks),ptr(error),rows,nodes,chunk,int64_t(1)),"pack selected SwiGLU Full actions");},
    {input.coordinates,input.valid,kinds,cursor,source,owners,destination,branch,chunks,error});
  p.branch(branch,{done,body});p.mark(body);
  p.index_select(mapping_,0,owners,parameters);
  p.index_select(comparisons,0,source,x.reshape({chunk,width}));p.index_select(contents,0,source,h.reshape({chunk,width}));
  p.index_select(gate_,0,parameters,gate);p.index_select(up_,0,parameters,up);p.index_select(down_,0,parameters,down);
  p.batch_matmul(x,gate,a);p.batch_matmul(x,up,b);p.silu(a,activated);p.multiply(activated,b,product);
  p.batch_matmul(product,down,result);p.add(result,h);p.index_copy(output,0,destination,result.reshape({chunk,width}));
  p.branch(branch,{head});p.mark(done);return {input.coordinates,output.narrow(0,0,rows),input.valid};
}
} // namespace tide::device_online
