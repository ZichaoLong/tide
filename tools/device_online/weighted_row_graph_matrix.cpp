#include "weighted_row_graph.h"
namespace tide::device_online {
size_t WeightedRowGraph::divide(size_t a,size_t b){
  const auto aw=value(a).size(1),bw=value(b).size(1);
  if(aw!=bw&&bw!=1)throw std::invalid_argument("row division requires equal or scalar denominator");
  auto out=at::zeros_like(value(a));program_->divide(value(a),value(b),out);return push({WeightedRowOp::Divide,out,{a,b}});
}
size_t WeightedRowGraph::matmul(size_t a,size_t b,int64_t m,int64_t k,int64_t n){
  if(m<1||k<1||n<1||m>value(a).size(1)/k||m*k!=value(a).size(1)||k>value(b).size(1)/n||k*n!=value(b).size(1))
    throw std::invalid_argument("row matrix product shape differs");
  if(4.L*rows_*m*n>budget_)throw std::invalid_argument("row matrix product output budget exceeded");
  auto out=at::zeros({rows_,m,n},value(a).options());program_->batch_matmul(value(a).view({rows_,m,k}),value(b).view({rows_,k,n}),out);
  WeightedRowExpression e{WeightedRowOp::Matmul,out.view({rows_,m*n}),{a,b}};e.dimensions={m,k,n};return push(e);
}
size_t WeightedRowGraph::convolution(size_t current,size_t cache,const WeightedRawLayout& l,const at::Tensor& integer,
    const at::Tensor& times,const WeightedConvParameters& w,int64_t stride,const std::string& role){
  const auto& field=l.field("conv_values");if(w.weight.size(0)!=banks_||value(cache).size(1)!=field.width*l.capacity())
    throw std::invalid_argument("row convolution cache/bank shape differs");
  auto raw=at::zeros({rows_,l.floating_width()},value(current).options());
  auto columns=at::arange(field.offset,field.offset+field.width*l.capacity(),owners_.options());
  program_->zero(raw);program_->index_copy(raw,1,columns,value(cache));
  auto t=append_weighted_convolution(*program_,l,{raw,integer},times,value(current),owners_,w,stride,error_,budget_);
  WeightedRowExpression e{WeightedRowOp::Convolution,t.output,{current,cache},{columns}};e.convolution=t;e.role=role;return push(e);
}
}
