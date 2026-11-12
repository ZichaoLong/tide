#pragma once
// Role matrices act on all node/sample/time rows without intermediate publication.
#include "weighted_kv_tensor.h"
#include "tide/packing_work.h"
namespace tide::weighted::columns::tensor {
struct Algebra {
  const Packed& p;
  explicit Algebra(const Packed& packed):p(packed){}
  const Tensor& at(const std::string& name)const{return p.values.at(name);}
  Tensor bias(const std::string& name)const{
    auto x=at(name);if(p.shared.count(name))return x;
    auto shape=x.sizes().vec();shape.insert(shape.begin()+1,2,1);return x.reshape(shape);
  }
  Tensor project(const Tensor& x,const std::string& name)const{
    const auto& w=at(name);const bool shared=p.shared.count(name);
    work::packing::projection(shared?"kv_tensor_shared":"kv_tensor",x.numel()/x.size(-1));
    if(shared)return at::matmul(x,w.t());
    auto shape=x.sizes().vec();shape.back()=w.size(1);
    return at::bmm(x.reshape({p.nodes,-1,x.size(-1)}),w.transpose(1,2)).reshape(shape);
  }
};
}
