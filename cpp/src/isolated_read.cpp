#include "tide/isolated_read.h"
#include "tide/operator_work.h"
#include <torch/csrc/autograd/custom_function.h>
#include <stdexcept>

namespace tide {
namespace {
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;
class IsolatedRead final : public torch::autograd::Function<IsolatedRead> {
 public:
  static variable_list forward(AutogradContext* ctx, at::TensorList rows, Tensor weight,
                                bool norm, int64_t dtype, at::Device device) {
    ctx->set_materialize_grads(false);
    auto x = at::stack(rows).to(device, static_cast<at::ScalarType>(dtype));
    auto w = norm ? Tensor() : weight.to(x.options());
    auto y = norm ? at::norm(x, 2, {-1}, false) : (x*w).sum(-1);
    auto outputs = y.unbind();
    variable_list frozen;
    for (size_t i=0;i<rows.size();++i)
      if (!rows[i].requires_grad() && (norm || !weight.requires_grad())) frozen.push_back(outputs[i]);
    ctx->mark_non_differentiable(frozen);
    ctx->saved_data["norm"] = norm;
    ctx->saved_data["input_device"] = rows[0].device();
    ctx->saved_data["input_dtype"] = int64_t(rows[0].scalar_type());
    variable_list saved{x,w,y,weight};
    saved.insert(saved.end(),rows.begin(),rows.end()); // Preserve input version checks.
    ctx->save_for_backward(saved);
    return outputs;
  }
  static variable_list backward(AutogradContext* ctx, variable_list grads) {
    if (at::GradMode::is_enabled()) throw std::invalid_argument("isolated Read supports first-order VJP only");
    variable_list result(grads.size()+4), bars;
    std::vector<Index> used;
    bool need_rows=false;
    for (size_t i=0;i<grads.size();++i) if (grads[i].defined()) {
      used.push_back(i); bars.push_back(grads[i]); need_rows |= ctx->needs_input_grad(i);
    }
    if (used.empty()) return result;
    const auto saved=ctx->get_saved_variables();
    auto ids=at::tensor(used,saved[0].options().dtype(at::kLong));
    auto dy=at::stack(bars).unsqueeze(-1);
    auto x=saved[0].index_select(0,ids);
    const bool norm=ctx->saved_data["norm"].toBool();
    if (need_rows) {
      Tensor dx;
      if (norm) {
        auto length=saved[2].index_select(0,ids).unsqueeze(-1);
        dx=(x*(dy/length.masked_fill(length==0,1))).masked_fill(length==0,0);
      } else dx=dy*saved[1];
      dx=dx.to(ctx->saved_data["input_device"].toDevice(),
               static_cast<at::ScalarType>(ctx->saved_data["input_dtype"].toInt()));
      auto rows=dx.unbind();
      for (size_t j=0;j<used.size();++j) if (ctx->needs_input_grad(used[j])) result[used[j]]=rows[j];
    }
    if (!norm && ctx->needs_input_grad(grads.size()))
      result[grads.size()]=(dy*x).sum(0).to(saved[3].options());
    return result;
  }
};
}
std::vector<Tensor> isolated_read(const std::vector<Tensor>& rows,const Tensor& weight,
                                  bool norm,at::ScalarType dtype,at::Device device) {
  if (rows.empty()) return {};
  work::StateReplayTimer timer(work::ReadBatchVjpNs);
  return IsolatedRead::apply(at::TensorList(rows),weight,norm,int64_t(dtype),device);
}
}
