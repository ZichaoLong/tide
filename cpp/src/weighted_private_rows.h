#pragma once
// Packed private leaves inside homogeneous query/region VJPs. Outer row/field
// roots, undefined cotangent groups and SavedVariable guards stay independent.
#include "tide/types.h"
#include "tide/packing_work.h"
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/variable.h>

namespace tide::weighted::private_rows {
inline Tensor join(const std::vector<Tensor>& values,bool stack=false){
  if(values.size()>1&&values[0].is_view()){
    auto base=Tensor(values[0]._base());Index cursor=values[0].storage_offset();bool valid=base.is_contiguous(),contiguous=true;
    for(const auto& t:values){
      valid=valid&&t.is_view()&&t._base().unsafeGetTensorImpl()==base.unsafeGetTensorImpl()
        &&t.is_contiguous()&&!t.is_conj()&&!t.is_neg()&&t.requires_grad()==base.requires_grad();
      contiguous=contiguous&&t.storage_offset()==cursor;cursor+=t.numel();
      if(stack)valid=valid&&t.sizes()==values[0].sizes();
      else valid=valid&&t.dim()==values[0].dim()&&t.sizes().slice(1)==values[0].sizes().slice(1);
    }
    if(valid){
      auto shape=values[0].sizes().vec();
      if(stack)shape.insert(shape.begin(),values.size());
      else {shape[0]=0;for(const auto& t:values)shape[0]+=t.size(0);}
      if(contiguous)return base.reshape({-1}).narrow(0,values[0].storage_offset()-base.storage_offset(),
                                     cursor-values[0].storage_offset()).reshape(shape);
      Index unit=stack?values[0].numel():1;
      if(!stack)for(Index d=1;d<values[0].dim();++d)unit*=values[0].size(d);
      valid=unit&&base.numel()%unit==0;
      if(valid)for(const auto& t:values)valid=valid&&(t.storage_offset()-base.storage_offset())%unit==0&&t.numel()%unit==0;
      if(valid){
        std::vector<Index> ids;
        for(const auto& t:values){auto start=(t.storage_offset()-base.storage_offset())/unit;
          for(Index i=0;i<t.numel()/unit;++i)ids.push_back(start+i);}
        work::packing::projection("private_root_gather",values.size());
        return base.reshape({-1,unit}).index_select(0,at::tensor(ids,base.options().dtype(at::kLong))).reshape(shape);
      }
    }
  }
  return stack?at::stack(values):at::cat(values);
}

struct Packing {
  size_t weights=0,rows=0,fields=0;
  std::vector<std::vector<std::pair<Index,Index>>> spans;
  std::vector<std::vector<Index>> shapes;
  std::vector<Tensor> pack(const std::vector<Tensor>& inputs,const std::vector<Tensor>& parameters,size_t count){
    weights=parameters.size();rows=count;fields=(inputs.size()-weights)/rows;
    std::vector<Tensor> leaves=parameters;
    for(size_t i=weights;i<inputs.size();++i)shapes.push_back(inputs[i].sizes().vec());
    at::NoGradGuard guard;
    for(size_t f=0;f<fields;++f){
      std::vector<Tensor> column;std::vector<std::pair<Index,Index>> offsets;Index cursor=0;
      for(size_t row=0;row<rows;++row){
        const auto& t=inputs[weights+row*fields+f];
        column.push_back(t.detach().reshape({-1}));offsets.push_back({cursor,t.numel()});cursor+=t.numel();
      }
      spans.push_back(std::move(offsets));
      leaves.push_back((rows==1?column[0]:at::cat(column)).set_requires_grad(inputs[weights+f].requires_grad()));
      work::packing::projection("private_root_column",rows);
    }
    return leaves;
  }
  std::vector<Tensor> views(const std::vector<Tensor>& leaves)const{
    std::vector<Tensor> result;
    for(size_t row=0;row<rows;++row)for(size_t f=0;f<fields;++f){
      auto [start,length]=spans[f][row];
      result.push_back(leaves[weights+f].narrow(0,start,length).reshape(shapes[row*fields+f]));
    }
    return result;
  }
  void scatter(size_t leaf,const std::vector<size_t>& used,const Tensor& gradient,std::vector<Tensor>& result)const{
    if(!gradient.defined())return;
    if(leaf<weights){auto& out=result[leaf];out=out.defined()?out+gradient:gradient;return;}
    auto f=leaf-weights;
    for(auto row:used){
      auto [start,length]=spans[f][row];auto value=gradient.narrow(0,start,length).reshape(shapes[row*fields+f]);
      auto& out=result[weights+row*fields+f];out=out.defined()?out+value:value;
    }
  }
};
}
