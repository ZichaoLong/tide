// Fixed-time convolution: one source use per lag, ordered bulk first-order VJP.
#include "weighted_column_layout.h"
#include <torch/csrc/autograd/custom_function.h>
namespace tide::weighted::columns {
namespace {
class TapGather final:public torch::autograd::Function<TapGather> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,const Tensor& rows,const Tensor& ids,const Tensor& reverse){
    ctx->set_materialize_grads(false);ctx->save_for_backward({rows,reverse});return gather(rows,ids);
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list bars){
    if(at::GradMode::is_enabled())throw std::invalid_argument("convolution tap gather supports first-order VJP only");
    auto saved=ctx->get_saved_variables();
    if(!bars[0].defined())return {Tensor{},Tensor{},Tensor{}};
    return {gather(bars[0].flatten(0,1),saved[1]).sum(1),Tensor{},Tensor{}};
  }
};
}
Tensor gather_taps(const Tensor& rows,const std::vector<Index>& ids,Index queries,Index taps){
  detail::require(taps>0&&queries>=0&&queries<=std::numeric_limits<Index>::max()/taps
    &&ids.size()==static_cast<size_t>(queries*taps),"invalid convolution tap layout");
  for(auto i:ids)detail::require(i>=0&&i<=rows.size(0),"convolution tap outside bank/sentinel");
  auto index=indices(ids,rows.device()).reshape({queries,taps});
  if(!at::GradMode::is_enabled()||!rows.requires_grad())return gather(rows,index);
  detail::require(rows.size(0)<=std::numeric_limits<Index>::max()/taps,"convolution transpose layout overflow");
  std::vector<Index> inverse(rows.size(0)*taps,queries*taps);
  for(Index j=0;j<queries*taps;++j)if(ids[j]<rows.size(0)){
    auto slot=ids[j]*taps+j%taps;
    detail::require(inverse[slot]==queries*taps,"convolution tap selects a bank row twice at the same lag");
    inverse[slot]=j;
  }
  return TapGather::apply(rows,index,indices(inverse,rows.device()).reshape({rows.size(0),taps}));
}
}
