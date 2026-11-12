#pragma once
// Readonly actual-row field owners. Public column roots share storage, not
// gradient identity; borrowing their base requires the explicit linear VJP.
#include "tide/types.h"
#include <algorithm>
#include "tide/packing_work.h"
#include <algorithm>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/variable.h>

namespace tide::weighted::columns::blocks {
struct FieldBlock {
  Tensor value;
  std::vector<Index> lengths;
  static FieldBlock publish(const Tensor& field,const Tensor& indices,const std::vector<Index>& lengths){
    work::packing::projection("column_block_publish",lengths.size());
    return {field.flatten(0,2).index_select(0,indices).detach(),lengths};
  }
  std::vector<Tensor> columns()const{
    std::vector<Tensor> result;Index offset=0;
    for(auto length:lengths){result.push_back(value.narrow(0,offset,length).detach());offset+=length;}
    return result;
  }
};

class Borrow final:public torch::autograd::Node {
 public:
  std::vector<torch::autograd::SavedVariable> originals;
  std::vector<Index> lengths;
  torch::autograd::variable_list apply(torch::autograd::variable_list&& bars)override{
    if(at::GradMode::is_enabled())throw std::invalid_argument("column block borrowing supports first-order VJP only");
    if(originals.empty())throw std::runtime_error("column block borrowing graph released");
    for(auto& value:originals)value.unpack();
    torch::autograd::variable_list result(lengths.size());
    if(!bars[0].defined())return result;
    Index offset=0;
    for(size_t i=0;i<lengths.size();++i){
      if(should_compute_output(i))result[i]=bars[0].narrow(0,offset,lengths[i]);
      offset+=lengths[i];
    }
    return result;
  }
  void release_variables()override{originals.clear();}
  std::string name()const override{return "tide::weighted::ColumnBlockBorrowVjp";}
};

inline Tensor concatenate(const std::vector<Tensor>& values){
  if(values.empty())throw std::invalid_argument("column concatenation needs at least one root");
  if(values.size()==1)return values[0];
  const auto& first=values[0];bool valid=first.dim()>0&&first.layout()==at::kStrided;
  const auto* storage=valid?first.storage().unsafeGetStorageImpl():nullptr;
  Index cursor=valid?first.storage_offset():0,total=0;bool grad=false;
  for(const auto& value:values){
    valid=valid&&value.dim()==first.dim()&&value.sizes().slice(1)==first.sizes().slice(1)
      &&value.layout()==at::kStrided&&value.device()==first.device()&&value.scalar_type()==first.scalar_type()
      &&value.is_contiguous()&&!value.is_conj()&&!value.is_neg()
      &&value.storage().unsafeGetStorageImpl()==storage&&value.storage_offset()==cursor;
    cursor+=value.numel();if(value.dim())total+=value.size(0);grad|=value.requires_grad();
  }
  if(!valid){work::packing::projection("column_block_copy",values.size());return at::cat(values);}
  work::packing::projection("column_block_borrow",values.size());
  auto shape=first.sizes().vec();shape[0]=total;Tensor result;
  // is_contiguous permits arbitrary strides on empty/size-one axes. A larger
  // borrowed view must use the proven logical contiguous span, not first's strides.
  std::vector<Index> strides(shape.size());Index stride=1;
  for(size_t i=shape.size();i>0;--i){strides[i-1]=stride;stride*=std::max<Index>(shape[i-1],1);}
  {at::NoGradGuard guard;result=first.as_strided(shape,strides).detach();}
  if(at::GradMode::is_enabled()&&grad){
    auto node=std::make_shared<Borrow>();
    node->set_next_edges(torch::autograd::collect_next_edges(values));
    for(const auto& value:values){node->originals.emplace_back(value,false);node->lengths.push_back(value.size(0));}
    torch::autograd::set_history(result,node);
  }
  return result;
}

inline Tensor stack_columns(const std::vector<Tensor>& values){
  if(values.empty())throw std::invalid_argument("column stack needs at least one root");
  for(const auto& value:values)if(value.sizes()!=values[0].sizes())throw std::invalid_argument("column stack shape mismatch");
  auto shape=values[0].sizes().vec();shape.insert(shape.begin(),values.size());
  return concatenate(values).reshape(shape).transpose(0,1);
}
}
