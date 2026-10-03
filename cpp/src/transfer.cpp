#include "tide/transfer.h"
#include "tide/device.h"
#include <torch/csrc/autograd/custom_function.h>
#include <algorithm>
#include <stdexcept>

namespace tide {
namespace {
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;
class CopyRows final : public torch::autograd::Function<CopyRows> {
 public:
  static variable_list forward(AutogradContext* ctx, at::TensorList rows, at::Device destination) {
    ctx->set_materialize_grads(false);
    ctx->saved_data["source"]=rows[0].device().str();
    auto outputs=at::stack(rows).to(destination).unbind(0);
    variable_list frozen;
    for(size_t i=0;i<rows.size();++i)if(!rows[i].requires_grad())frozen.push_back(outputs[i]);
    ctx->mark_non_differentiable(frozen);
    return outputs;
  }
  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    variable_list result(grads.size()+1),used;
    std::vector<size_t> ids;
    for(size_t i=0;i<grads.size();++i)if(grads[i].defined()&&ctx->needs_input_grad(i)) {
      ids.push_back(i);used.push_back(grads[i]);
    }
    if(!used.empty()) {
      auto values=copy_rows(used,at::Device(ctx->saved_data["source"].toStringRef()));
      for(size_t i=0;i<ids.size();++i)result[ids[i]]=values[i];
    }
    return result;
  }
};
}
std::vector<Tensor> copy_rows(const std::vector<Tensor>& rows,at::Device destination,Index budget) {
  if(budget<1)throw std::invalid_argument("transfer packing budget must be positive");
  if(rows.empty())return {};
  const auto& first=rows[0];
  if(!supported_kernel_payload(first)||first.dim()!=1||first.numel()==0)
    throw std::invalid_argument("transfer requires nonempty FP16/FP32/FP64 vector rows");
  for(const auto& row:rows)
    if(!row.defined()||row.layout()!=at::kStrided||row.sizes()!=first.sizes()
       ||row.device()!=first.device()||row.scalar_type()!=first.scalar_type())
      throw std::invalid_argument("transfer row metadata mismatch");
  const auto limit=std::max<Index>(1,budget/(first.numel()*first.element_size()));
  std::vector<Tensor> result;result.reserve(rows.size());
  for(size_t start=0;start<rows.size();) {
    const auto count=std::min<size_t>(limit,rows.size()-start);
    if(count==1)result.push_back(rows[start].to(destination));
    else {
      auto values=CopyRows::apply(at::TensorList(rows.data()+start,count),destination);
      result.insert(result.end(),values.begin(),values.end());
    }
    start+=count;
  }
  return result;
}
} // namespace tide
