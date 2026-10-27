#include "weighted_row_graph.h"
#include "device_launch_tide_weighted_row_bind.h"
#include <ATen/core/grad_mode.h>
#include <cmath>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void dimensions(const at::Tensor& x,const at::Tensor& owners){
  if(!x.defined()||x.dim()!=2||x.size(0)!=owners.numel()||x.size(1)<1||!x.is_contiguous()||x.requires_grad()
    ||x.scalar_type()!=at::kFloat||x.device()!=owners.device())throw std::invalid_argument("invalid local row expression input");
}
}
WeightedRowGraph::WeightedRowGraph(DeviceProgram& p,const at::Tensor& owners,int64_t banks,const at::Tensor& error,int64_t available)
    :program_(&p),error_(error),rows_(owners.numel()),banks_(banks),budget_(available){
  if(at::GradMode::is_enabled()||!owners.defined()||owners.dim()!=1||rows_<1||banks<1||available<1
    ||owners.scalar_type()!=at::kLong||!owners.is_contiguous()||owners.device().type()!=resident_device_type
    ||!error.defined()||error.sizes()!=at::IntArrayRef{1}||error.scalar_type()!=at::kInt||error.device()!=owners.device())
    throw std::invalid_argument("local row graph requires explicit bounded accelerator owners");
  validate_kernel_device(owners.device());initial_bytes_=p.retained_tensor_bytes();owners_=at::zeros_like(owners);p.copy(owners_,owners);
}
size_t WeightedRowGraph::push(WeightedRowExpression e){
  dimensions(e.value,owners_);retained_+=e.value.numel()*4.L;
  auto owners=owners_,error=error_,out=e.value;const auto banks=banks_;
  program_->kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_row_bind)(1,stream,
    ptr(owners),ptr(out),ptr(out),ptr(error),out.size(0),out.size(1),banks,0),"local complete row result guard");},{owners,out,error});
  if(retained_>budget_||program_->retained_tensor_bytes()-initial_bytes_>budget_)throw std::invalid_argument("local row graph tensor budget exceeded");
  expressions_.push_back(std::move(e));return expressions_.size()-1;
}
const at::Tensor& WeightedRowGraph::value(size_t index)const{return expressions_.at(index).value;}
size_t WeightedRowGraph::input(const at::Tensor& x,bool positive){
  dimensions(x,owners_);auto out=at::zeros_like(x);auto owners=owners_,error=error_;const auto banks=banks_;
  program_->kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_row_bind)(1,stream,
    ptr(owners),ptr(x),ptr(out),ptr(error),x.size(0),x.size(1),banks,positive),"local actual row inputs");},{owners,x,out,error});
  return push({WeightedRowOp::Input,out});
}
size_t WeightedRowGraph::constant(int64_t width,double scalar){
  if(width<1||!std::isfinite(scalar)||!std::isfinite(float(scalar)))throw std::invalid_argument("invalid local row constant");
  auto source=at::full({rows_,width},float(scalar),owners_.options().dtype(at::kFloat)),out=at::zeros_like(source);
  program_->copy(out,source);return push({WeightedRowOp::Constant,out});
}
size_t WeightedRowGraph::projection(size_t a,const WeightedProjectionParameters& w,const std::string& role){
  if(w.matrix.size(0)!=banks_)throw std::invalid_argument("row projection bank count differs");
  auto t=append_weighted_projection(*program_,value(a),owners_,w,std::nullopt,error_,budget_);
  WeightedRowExpression e{WeightedRowOp::Projection,t.output,{a}};e.role=role;e.projection=t;return push(e);
}
size_t WeightedRowGraph::head(size_t a,const WeightedHeadParameters& w,const std::string& role){
  if(w.matrix.size(0)!=banks_)throw std::invalid_argument("row head bank count differs");
  auto t=append_weighted_head(*program_,value(a),owners_,w,error_,budget_);WeightedRowExpression e{WeightedRowOp::Head,t.output,{a}};e.role=role;e.head=t;return push(e);
}
size_t WeightedRowGraph::norm(size_t a,double epsilon){
  if(!std::isfinite(epsilon)||!std::isfinite(float(epsilon))||float(epsilon)<=0)throw std::invalid_argument("invalid row normalization epsilon");
  auto out=at::zeros_like(value(a)),rstd=at::zeros({rows_,1},out.options());program_->rms_norm(value(a),epsilon,out,rstd);
  WeightedRowExpression e{WeightedRowOp::Norm,out,{a}};e.rstd=rstd;return push(e);
}
size_t WeightedRowGraph::cat(const std::vector<size_t>& input){
  if(input.empty())throw std::invalid_argument("empty row concatenate");int64_t width=0;for(auto i:input)width+=value(i).size(1);
  auto out=at::zeros({rows_,width},value(input[0]).options());WeightedRowExpression e{WeightedRowOp::Cat,out,input};
  int64_t start=0;for(auto i:input){auto indices=at::arange(start,start+value(i).size(1),owners_.options());program_->index_copy(out,1,indices,value(i));e.indices.push_back(indices);start+=value(i).size(1);}return push(e);
}
size_t WeightedRowGraph::slice(size_t a,int64_t start,int64_t count){
  if(start<0||count<1||start>value(a).size(1)-count)throw std::invalid_argument("invalid local row slice");
  auto out=at::zeros({rows_,count},value(a).options()),indices=at::arange(start,start+count,owners_.options());
  program_->index_select(value(a),1,indices,out);return push({WeightedRowOp::Slice,out,{a},{indices}});
}
size_t WeightedRowGraph::add(size_t a,size_t b){
  if(value(a).sizes()!=value(b).sizes())throw std::invalid_argument("row addition shape differs");
  auto out=at::zeros_like(value(a));program_->copy(out,value(a));program_->add(out,value(b));return push({WeightedRowOp::Add,out,{a,b}});
}
size_t WeightedRowGraph::product(size_t a,size_t b){
  auto aw=value(a).size(1),bw=value(b).size(1);if(aw!=bw&&aw!=1&&bw!=1)throw std::invalid_argument("invalid local row product broadcast");
  auto out=at::zeros({rows_,std::max(aw,bw)},value(a).options());program_->multiply(value(a),value(b),out);return push({WeightedRowOp::Product,out,{a,b}});
}
size_t WeightedRowGraph::scale(size_t a,double factor,double offset){
  if(!std::isfinite(factor)||!std::isfinite(float(factor))||!std::isfinite(offset)||!std::isfinite(float(offset)))throw std::invalid_argument("invalid local row affine scalar");
  auto out=at::zeros_like(value(a));program_->multiply(value(a),at::full({1},float(factor),out.options()),out);
  if(offset!=0)program_->add(out,at::full(out.sizes(),float(offset),out.options()));WeightedRowExpression e{WeightedRowOp::Scale,out,{a}};e.scalar=factor;return push(e);
}
size_t WeightedRowGraph::sum(size_t a){auto out=at::zeros({rows_,1},value(a).options());program_->sum(value(a),1,true,out);return push({WeightedRowOp::Sum,out,{a}});}
size_t WeightedRowGraph::unary(WeightedRowOp kind,size_t a,double scalar){
  auto out=at::zeros_like(value(a));switch(kind){
    case WeightedRowOp::Sigmoid:program_->sigmoid(value(a),out);break;
    case WeightedRowOp::Silu:program_->silu(value(a),out);break;
    case WeightedRowOp::Log1p:program_->log1p(value(a),out);break;
    case WeightedRowOp::Exp:program_->exp(value(a),out);break;
    case WeightedRowOp::Sqrt:program_->sqrt(value(a),out);break;
    case WeightedRowOp::Softplus:program_->softplus(value(a),out);break;
    case WeightedRowOp::Clamp:program_->minimum(value(a),at::full({1},float(scalar),out.options()),out);break;
    default:throw std::logic_error("invalid row unary operation");}
  WeightedRowExpression e{kind,out,{a}};e.scalar=scalar;return push(e);
}
size_t WeightedRowGraph::sigmoid(size_t a){return unary(WeightedRowOp::Sigmoid,a);}
size_t WeightedRowGraph::silu(size_t a){return unary(WeightedRowOp::Silu,a);}
size_t WeightedRowGraph::log1p(size_t a){return unary(WeightedRowOp::Log1p,a);}
size_t WeightedRowGraph::exp(size_t a){return unary(WeightedRowOp::Exp,a);}
size_t WeightedRowGraph::sqrt(size_t a){return unary(WeightedRowOp::Sqrt,a);}
size_t WeightedRowGraph::softplus(size_t a){return unary(WeightedRowOp::Softplus,a);}
size_t WeightedRowGraph::clamp_max(size_t a,double bound){if(!std::isfinite(bound)||!std::isfinite(float(bound)))throw std::invalid_argument("invalid row clamp bound");return unary(WeightedRowOp::Clamp,a,bound);}
}
