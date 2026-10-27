#include "weighted_row_graph.h"
#include "device_launch_tide_weighted_row_accumulate.h"
#include "device_launch_tide_weighted_row_derivative.h"
#include "device_launch_tide_weighted_projection_norm.h"
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& t){return static_cast<uint8_t*>(t.data_ptr());}
void valid(const at::Tensor& x,at::Device device,at::ScalarType type,at::IntArrayRef shape){
  if(!x.defined()||x.device()!=device||x.scalar_type()!=type||x.sizes()!=shape||!x.is_contiguous()||x.requires_grad())
    throw std::invalid_argument("invalid local row cotangent");
}
}
WeightedRowReverse WeightedRowGraph::reverse(DeviceProgram& p,const std::vector<WeightedRowRoot>& roots,
    const at::Tensor& error,int64_t available)const{
  if(available<1||retained_*4>available)throw std::invalid_argument("row reverse tensor budget exceeded");
  const auto initial=p.retained_tensor_bytes();valid(error,owners_.device(),at::kInt,{1});WeightedRowReverse result;
  for(const auto& e:expressions_){auto bar=at::zeros_like(e.value),flag=at::zeros({rows_},owners_.options().dtype(at::kBool));
    p.zero(bar);p.zero(flag);result.rows.push_back({bar,flag});}
  const auto owners=owners_;
  auto accumulate=[&](size_t index,const at::Tensor& incoming,const at::Tensor& connected){
    auto& target=result.rows.at(index);valid(incoming,owners.device(),at::kFloat,target.value.sizes());valid(connected,owners.device(),at::kBool,{rows_});
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_row_accumulate)(1,stream,
      ptr(owners),ptr(incoming),ptr(connected),ptr(target.value),ptr(target.connected),ptr(error),incoming.size(0),incoming.size(1)),
      "local row cotangent and structural contribution");},{owners,incoming,connected,target.value,target.connected,error});
  };
  auto parameter=[&](const std::string& name,const at::Tensor& value,const at::Tensor& connected){
    if(!result.parameters.emplace(name,WeightedRowAdjoint{value,connected}).second)throw std::invalid_argument("duplicate local parameter role");};
  for(const auto& r:roots)accumulate(r.index,r.value,r.connected);
  for(size_t cursor=expressions_.size();cursor>0;--cursor){const auto index=cursor-1;const auto& e=expressions_[index];const auto& bar=result.rows[index];
    if(e.kind==WeightedRowOp::Input||e.kind==WeightedRowOp::Constant)continue;
    const auto a=e.parents[0];const auto& x=value(a);
    switch(e.kind){
      case WeightedRowOp::Projection:{auto g=append_weighted_projection_vjp(p,*e.projection,bar.value,bar.connected,error,available);
        accumulate(a,g.input,g.row_connected);parameter(e.role+".matrix",g.parameters.matrix,g.owner_connected);
        if(g.parameters.bias.defined())parameter(e.role+".bias",g.parameters.bias,g.owner_connected);break;}
      case WeightedRowOp::Head:{auto g=append_weighted_head_vjp(p,*e.head,bar.value,bar.connected,error,available);
        accumulate(a,g.input,g.row_connected);parameter(e.role+".matrix",g.parameters.matrix,g.owner_connected);parameter(e.role+".hidden_bias",g.parameters.hidden_bias,g.owner_connected);
        parameter(e.role+".projection",g.parameters.projection,g.owner_connected);parameter(e.role+".bias",g.parameters.bias,g.owner_connected);break;}
      case WeightedRowOp::Norm:{auto dx=at::zeros_like(x);p.copy(dx,bar.value);
        p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_projection_norm)(1,stream,
          ptr(e.value),ptr(e.rstd),ptr(dx),ptr(bar.connected),ptr(error),x.size(0),x.size(1)),"local saved normalization VJP");},
          {e.value,e.rstd,dx,bar.connected,error});accumulate(a,dx,bar.connected);break;}
      case WeightedRowOp::Cat:{for(size_t j=0;j<e.parents.size();++j){auto part=at::zeros_like(value(e.parents[j]));p.index_select(bar.value,1,e.indices[j],part);accumulate(e.parents[j],part,bar.connected);}break;}
      case WeightedRowOp::Slice:{auto dx=at::zeros_like(x);p.zero(dx);p.index_copy(dx,1,e.indices[0],bar.value);accumulate(a,dx,bar.connected);break;}
      case WeightedRowOp::Add:accumulate(a,bar.value,bar.connected);accumulate(e.parents[1],bar.value,bar.connected);break;
      case WeightedRowOp::Product:{for(int j=0;j<2;++j){auto dx=at::zeros_like(e.value);p.multiply(bar.value,value(e.parents[1-j]),dx);
          if(value(e.parents[j]).size(1)==1&&e.value.size(1)!=1){auto reduced=at::zeros_like(value(e.parents[j]));p.sum(dx,1,true,reduced);accumulate(e.parents[j],reduced,bar.connected);}
          else accumulate(e.parents[j],dx,bar.connected);}break;}
      case WeightedRowOp::Scale:{auto dx=at::zeros_like(x);p.multiply(bar.value,at::full({1},float(e.scalar),x.options()),dx);accumulate(a,dx,bar.connected);break;}
      case WeightedRowOp::Sum:{auto dx=at::zeros_like(x);p.multiply(bar.value,at::ones_like(x),dx);accumulate(a,dx,bar.connected);break;}
      case WeightedRowOp::Silu:{auto dx=at::zeros_like(x);p.silu_backward(bar.value,x,dx);accumulate(a,dx,bar.connected);break;}
      case WeightedRowOp::Sigmoid:case WeightedRowOp::Log1p:case WeightedRowOp::Clamp:{auto dx=at::zeros_like(x);
        const int64_t mode=e.kind==WeightedRowOp::Sigmoid?0:(e.kind==WeightedRowOp::Log1p?1:2);
        p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_row_derivative)(1,stream,
          ptr(x),ptr(e.value),ptr(bar.value),ptr(bar.connected),ptr(dx),ptr(error),x.size(0),x.size(1),mode,float(e.scalar)),
          "local declared scalar-map VJP");},{x,e.value,bar.value,bar.connected,dx,error});accumulate(a,dx,bar.connected);break;}
      default:throw std::logic_error("missing local row reverse mapping");
    }
    if(p.retained_tensor_bytes()-initial>available)throw std::invalid_argument("row reverse retained tensor budget exceeded");
  }
  return result;
}
}
