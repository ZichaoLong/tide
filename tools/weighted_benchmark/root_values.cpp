#include "root_values.h"
#include "tide/transfer.h"
#include <torch/csrc/autograd/autograd.h>
namespace weighted_benchmark {
namespace {
at::Tensor objective(const std::vector<at::Tensor>& values) {
  at::Tensor sum;
  for(const auto& value:values) {
    if(!value.defined()||!value.is_floating_point()||!value.numel())continue;
    auto x=value.scalar_type()==at::kHalf?value.to(at::kFloat):value;
    auto term=x.square().mean();
    if(sum.defined()&&term.device()!=sum.device())
      term=tide::copy_rows({term.reshape({1})},sum.device()).front().reshape({});
    sum=sum.defined()?sum+term:term;
  }
  return sum.defined()?sum:at::zeros({},at::TensorOptions().dtype(at::kDouble));
}
}
std::map<std::string,at::Tensor> diagnostic_roots(const RootDiagnostic& diagnostic) {
  const auto& r=diagnostic.result;
  std::vector<at::Tensor> all,output,control,descriptor;
  for(const auto& x:r.outputs)output.push_back(x.value);
  for(const auto& e:r.trace) {
    control.push_back(e.control);descriptor.push_back(e.descriptor);
    for(const auto& [_,x]:e.descriptor_fields)descriptor.push_back(x);
  }
  for(const auto* fields:{&output,&control,&descriptor})all.insert(all.end(),fields->begin(),fields->end());
  for(const auto& [_,s]:r.continuation.states) {
    all.push_back(s.value);for(const auto& [__,x]:s.slots)all.push_back(x);
  }
  for(const auto& [_,h]:r.continuation.history)for(const auto& [__,x]:h.tensors)all.push_back(x);
  for(const auto& x:r.continuation.pending)all.push_back(x.value);
  for(const auto& x:r.messages)all.push_back(x.value);
  return {{"all",objective(all)},{"output",objective(output)},
          {"control",objective(control)},{"descriptor",objective(descriptor)}};
}
std::map<std::string,at::Tensor> diagnostic_vjp(const at::Tensor& root,
    const std::map<std::string,at::Tensor>& leaves,bool zero) {
  std::map<std::string,at::Tensor> result;
  std::vector<at::Tensor> values;
  for(const auto& [name,value]:leaves) {
    if(!value.defined()||!value.requires_grad())throw std::invalid_argument("diagnostic leaf requires grad: "+name);
    result[name]=at::Tensor{};values.push_back(value);
  }
  if(!root.requires_grad()||values.empty())return result;
  // A connected zero cotangent must stay distinguishable from a disconnected leaf.
  const auto gradients=torch::autograd::grad({root*(zero?0.:.7)},values,{},true,false,true);
  size_t index=0;for(auto& [_,value]:result)value=gradients.at(index++);
  return result;
}
}  // namespace weighted_benchmark
