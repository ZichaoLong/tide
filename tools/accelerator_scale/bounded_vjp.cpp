#include "bounded.h"
#include "peer_transport.h"
#include <torch/csrc/autograd/autograd.h>
#include <stdexcept>

namespace accelerator_scale::bounded {
Value row(const Value& value,Index batch) { return {value.data[batch],value.dependencies[batch].unsqueeze(0)}; }
Value root(const Tensor& tensor,const Tensor& dependencies) { return {tensor,dependencies.any(0).unsqueeze(0)}; }
Value Program::loss(const Window& window,const Tensor& ids) const {
  std::vector<Tensor> losses;auto dep=empty_dependencies(f_.head.device());
  for(Index token=0;token<limits_.tokens;++token) {
    const auto& v=window.logits.at(token);
    auto target=(move(ids[token],v.data.device())+1).remainder(f_.head.size(0));
    losses.push_back(at::cross_entropy_loss(v.data.to(at::kFloat),target));dep=dep|v.dependencies;
  }
  return root(at::stack(losses).mean(),dep);
}
std::vector<Tensor> Program::vjp(const Value& value,const Tensor& cotangent,bool retain,
                                const std::vector<Tensor>& external_roots) const {
  if(!limits_.connectivity)throw std::invalid_argument("VJP requires structural connectivity tracking");
  auto leaves=leaves_;leaves.insert(leaves.end(),external_roots.begin(),external_roots.end());
  if(!value.data.requires_grad())return std::vector<Tensor>(leaves.size());
  if(auto peer=replay_peer_vjp(value.data,leaves,cotangent,retain))return *peer;
  return torch::autograd::grad({value.data},leaves,{cotangent},retain,false,true);
}
std::vector<Tensor> export_gradients(const Value& value,std::vector<Tensor> gradients) {
  auto bits=value.dependencies.any(0).to(at::kCPU).contiguous();
  if(bits.numel()<Index(gradients.size()))throw std::logic_error("connectivity leaf inventory");
  const auto* present=bits.const_data_ptr<bool>();
  for(size_t i=0;i<gradients.size();++i) {
    if(!present[i]) {
      if(gradients[i].defined() && at::count_nonzero(gradients[i]).item<int64_t>()!=0)
        throw std::runtime_error("nonzero derivative for structurally disconnected leaf "+std::to_string(i));
      gradients[i]=Tensor();
    } else if(!gradients[i].defined())throw std::runtime_error("missing structurally connected derivative "+std::to_string(i));
  }
  return gradients;
}
}  // namespace accelerator_scale::bounded
