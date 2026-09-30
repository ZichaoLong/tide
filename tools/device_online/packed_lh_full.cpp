#include "packed_lh_full.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_full_plan.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <set>
#include <stdexcept>

namespace tide::device_online {
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
int64_t lh_full_kind(const std::string& name) {
  int64_t kind=0;
  for(const std::string act:{"relu","silu","identity"})for(const std::string norm:{"identity","rms","layer"}) {
    ++kind;if(name=="lh-"+act+"-"+norm+"-v1")return kind;
  }
  return 0;
}
PackedLhFull::PackedLhFull(std::vector<int64_t> kinds,const at::Tensor& weight,const at::Tensor& bias,
    at::Device device,int64_t capacity,int64_t max_rows,int64_t budget)
    :nodes_(kinds.size()),width_(weight.defined()&&weight.dim()==2?weight.size(1):0),rows_(capacity),chunk_(max_rows),reserved_(0) {
  if(at::GradMode::is_enabled()||nodes_<1||width_<1||capacity<1||max_rows<1||budget<1
      ||device.type()!=c10::DeviceType::PrivateUse1||!weight.device().is_cpu()||weight.scalar_type()!=at::kFloat
      ||weight.sizes()!=at::IntArrayRef{nodes_,width_}||bias.sizes()!=weight.sizes()||bias.device()!=weight.device()
      ||bias.scalar_type()!=at::kFloat||weight.requires_grad()||bias.requires_grad())
    throw std::invalid_argument("packed LH Full requires CPU FP32 affine values and no-grad NPU");
  std::set<int64_t> groups;
  for(auto kind:kinds){if(kind<0||kind>9)throw std::invalid_argument("unknown packed LH Full contract");if(kind)groups.insert(kind);}
  groups_.assign(groups.begin(),groups.end());
  // Include parameter sentinels, action copies and reusable chunk vectors.
  const long double fixed=8.L*(nodes_+1.L)*width_+8.L*(rows_+1.L)*width_+8.L*nodes_;
  const long double per_row=128.L*width_+128;
  if(fixed+per_row>budget)throw std::invalid_argument("one packed LH Full row exceeds workspace budget");
  chunk_=std::min<int64_t>({max_rows,capacity,static_cast<int64_t>((budget-fixed)/per_row)});
  reserved_=static_cast<int64_t>(fixed+per_row*chunk_);
  kinds_=at::tensor(kinds,at::kLong).to(device);
  weights_=at::cat({weight,at::zeros({1,width_},weight.options())},0).to(device);
  biases_=at::cat({bias,at::zeros({1,width_},bias.options())},0).to(device);
}
ActionBatch PackedLhFull::append_stage(CannProgram& p,const ActionBatch& input,const at::Tensor& comparison,
    const at::Tensor& error,const at::Tensor& chunks) {
  const auto rows=rows_,width=width_,chunk=chunk_,nodes=nodes_;
  if(input.values.sizes()!=at::IntArrayRef{rows,width}||comparison.sizes()!=input.values.sizes()
      ||input.values.scalar_type()!=at::kFloat||comparison.scalar_type()!=at::kFloat
      ||input.coordinates.sizes()!=at::IntArrayRef{rows,4}||input.coordinates.scalar_type()!=at::kLong
      ||input.valid.sizes()!=at::IntArrayRef{rows}||input.valid.scalar_type()!=at::kBool
      ||error.sizes()!=at::IntArrayRef{1}||error.scalar_type()!=at::kInt
      ||chunks.sizes()!=at::IntArrayRef{1}||chunks.scalar_type()!=at::kLong)
    throw std::invalid_argument("invalid LH Full action/metadata shapes");
  for(const auto& x:{input.values,comparison,input.coordinates,input.valid,error,chunks})
    if(x.device()!=kinds_.device()||!x.is_contiguous()||x.requires_grad())throw std::invalid_argument("invalid LH Full device/ownership");
  if(groups_.empty())return input;
  auto source=at::zeros({chunk},kinds_.options()),parameters=at::zeros_like(source),destination=at::zeros_like(source);
  auto cursor=at::zeros({1},kinds_.options()),zero=at::zeros_like(cursor),branch=at::zeros_like(error);
  auto comparisons=at::zeros({rows+1,width},input.values.options());
  auto output=at::zeros({rows+chunk,width},input.values.options());
  auto x=at::empty({chunk,width},input.values.options()),activated=at::empty_like(x),normalized=at::empty_like(x);
  auto weight=at::empty_like(x),bias=at::empty_like(x),result=at::empty_like(x);
  p.copy(comparisons.narrow(0,0,rows),comparison);p.copy(output.narrow(0,0,rows),input.values);
  const auto kinds=kinds_;
  for(const auto kind:groups_) {
    const auto act=(kind-1)/3,norm=(kind-1)%3;
    p.copy(cursor,zero);auto head=p.label(),body=p.label(),done=p.label();p.mark(head);
    p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_full_plan)(1,stream,
      ptr(input.coordinates),ptr(input.valid),ptr(kinds),ptr(cursor),ptr(source),ptr(parameters),ptr(destination),
      ptr(branch),ptr(chunks),ptr(error),rows,nodes,chunk,kind),"pack selected LH Full actions");},
      {input.coordinates,input.valid,kinds,cursor,source,parameters,destination,branch,chunks,error});
    p.branch(branch,{done,body});p.mark(body);p.index_select(comparisons,0,source,x);
    if(act==0)p.relu(x,activated);else if(act==1)p.silu(x,activated);else p.copy(activated,x);
    if(norm==1)p.rms_norm(activated,1e-7,normalized);
    else if(norm==2)p.layer_norm(activated,1e-5,normalized);
    else p.copy(normalized,activated);
    if(norm) {
      p.index_select(weights_,0,parameters,weight);p.multiply(normalized,weight,result);
      if(norm==2){p.index_select(biases_,0,parameters,bias);p.add(result,bias);}
    }else p.copy(result,normalized);
    p.index_copy(output,0,destination,result);p.branch(branch,{head});p.mark(done);
  }
  return {input.coordinates,output.narrow(0,0,rows),input.valid};
}
} // namespace tide::device_online
