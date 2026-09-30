#include "packed_full.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_full_plan.h"
#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
namespace {
std::pair<long double,long double> footprint(int64_t nodes,int64_t width) {
  return {4.L*(nodes+1.L)*(width*static_cast<long double>(width)+width),8.L*width*width+64.L*width+64};
}
}
long double PackedFull::minimum_bytes(const std::vector<int64_t>& kinds,int64_t width) {
  if(std::find(kinds.begin(),kinds.end(),1)==kinds.end())return 1;
  const auto [fixed,row]=footprint(kinds.size(),width);return fixed+row;
}
namespace {uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}}
PackedFull::PackedFull(std::vector<int64_t> kinds,const at::Tensor& weight,const at::Tensor& bias,at::Device device,int64_t max_rows,int64_t budget)
    :nodes_(kinds.size()),width_(bias.defined()&&bias.dim()==2?bias.size(1):0),chunk_(max_rows),any_tanh_(false) {
  if(at::GradMode::is_enabled()||nodes_<1||width_<1||max_rows<1||budget<1||device.type()!=c10::DeviceType::PrivateUse1||!bias.device().is_cpu()
      ||bias.scalar_type()!=at::kFloat||bias.sizes()!=at::IntArrayRef{nodes_,width_}
      ||weight.sizes()!=at::IntArrayRef{nodes_,width_,width_}||weight.device()!=bias.device()||weight.scalar_type()!=at::kFloat
      ||bias.requires_grad()||weight.requires_grad())throw std::invalid_argument("packed Full requires CPU FP32 parameters, NPU and no-grad");
  for(auto kind:kinds){if(kind<0||kind>1)throw std::invalid_argument("unknown packed Full contract");any_tanh_|=kind==1;}
  // Account for gathered matrices and vector work before selecting a chunk.
  const auto [persistent,per_row]=footprint(nodes_,width_);
  if(any_tanh_&&(persistent+per_row>budget))throw std::invalid_argument("one packed Full row exceeds workspace budget");
  if(any_tanh_)chunk_=std::min<int64_t>(max_rows,static_cast<int64_t>((budget-persistent)/per_row));
  else chunk_=1;
  if(any_tanh_)reserved_=static_cast<int64_t>(persistent+per_row*chunk_);
  kinds_=at::tensor(kinds,at::kLong).to(device);chunks_=at::zeros({1},kinds_.options());
  if(any_tanh_) {
    weights_=at::cat({weight,at::zeros({1,width_,width_},weight.options())},0).to(device).contiguous();
    biases_=at::cat({bias,at::zeros({1,width_},bias.options())},0).to(device).contiguous();
  }
}
ActionBatch PackedFull::append_stage(CannProgram& p,const ActionBatch& content,const at::Tensor& comparison,const at::Tensor& error) {
  if(!content.values.defined()||content.values.dim()!=2)throw std::invalid_argument("invalid packed Full value buffer");
  const auto rows=content.values.size(0),width=width_,chunk=chunk_,nodes=nodes_;
  if(rows<1||rows>std::numeric_limits<int64_t>::max()/4/width-chunk||content.values.size(1)!=width||comparison.sizes()!=content.values.sizes()
      ||content.coordinates.sizes()!=at::IntArrayRef{rows,4}||content.valid.sizes()!=at::IntArrayRef{rows}
      ||error.sizes()!=at::IntArrayRef{1}||error.scalar_type()!=at::kInt
      ||content.values.scalar_type()!=at::kFloat||comparison.scalar_type()!=at::kFloat
      ||content.coordinates.scalar_type()!=at::kLong||content.valid.scalar_type()!=at::kBool)
    throw std::invalid_argument("invalid packed Full action shapes");
  for(const auto& x:{content.values,comparison,content.coordinates,content.valid,error})
    if(x.device()!=kinds_.device()||!x.is_contiguous()||x.requires_grad())throw std::invalid_argument("invalid packed Full device/ownership");
  if(!any_tanh_)return content;
  auto source=at::zeros({chunk},kinds_.options()),parameters=at::zeros_like(source),destination=at::zeros_like(source);
  auto cursor=at::zeros({1},kinds_.options()),zero=at::zeros_like(cursor),branch=at::zeros_like(error);
  auto contents=at::zeros({rows+1,width},content.values.options()),comparisons=at::zeros_like(contents);
  auto output=at::zeros({rows+chunk,width},content.values.options());
  auto x=at::empty({chunk,width},content.values.options()),h=at::empty_like(x),bias=at::empty_like(x),activation=at::empty_like(x);
  auto weights=at::empty({chunk,width,width},content.values.options()),product=at::empty_like(x);
  p.copy(cursor,zero);p.copy(contents.narrow(0,0,rows),content.values);p.copy(comparisons.narrow(0,0,rows),comparison);
  p.copy(output.narrow(0,0,rows),content.values); // Identity Full is exactly content.
  const auto kinds=kinds_,chunks=chunks_;
  auto head=p.label(),body=p.label(),done=p.label();p.mark(head);
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_full_plan)(1,stream,ptr(content.coordinates),ptr(content.valid),
    ptr(kinds),ptr(cursor),ptr(source),ptr(parameters),ptr(destination),ptr(branch),ptr(chunks),ptr(error),rows,nodes,chunk,int64_t(1)),
    "pack selected Full actions");},{content.coordinates,content.valid,kinds,cursor,source,parameters,destination,branch,chunks,error});
  p.branch(branch,{done,body});p.mark(body);
  p.index_select(comparisons,0,source,x);p.index_select(contents,0,source,h);
  p.index_select(weights_,0,parameters,weights);p.index_select(biases_,0,parameters,bias);
  p.batch_matmul(x.reshape({chunk,1,width}),weights,product.reshape({chunk,1,width}));
  p.add(product,bias);p.tanh(product,activation);p.add(activation,h);
  // Real rows are distinct; padding gets a different scratch destination per
  // slot. There are no conflicting writes or arithmetic on inactive owners.
  p.index_copy(output,0,destination,activation);p.branch(branch,{head});p.mark(done);
  return {content.coordinates,output.narrow(0,0,rows),content.valid};
}
} // namespace tide::device_online
