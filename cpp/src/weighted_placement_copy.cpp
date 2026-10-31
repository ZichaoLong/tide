// Cast/transfer is an explicit first-order edge and preserves original versions.
#include "weighted_placement.h"
#include <torch/csrc/autograd/custom_function.h>
#include <sstream>
namespace tide::weighted {
namespace {
class Copy final:public torch::autograd::Function<Copy>{
 public:
  static Tensor forward(torch::autograd::AutogradContext* ctx,Tensor value,at::Device device,at::ScalarType dtype){
    ctx->set_materialize_grads(false);ctx->save_for_backward({value});
    return value.to(value.options().device(device).dtype(dtype));
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list bars){
    if(at::GradMode::is_enabled())throw std::invalid_argument("weighted placement supports first-order VJP only");
    auto value=ctx->get_saved_variables()[0];
    return {bars[0].defined()?bars[0].to(value.options()):Tensor{},Tensor{},Tensor{}};
  }
};
class Copies final:public torch::autograd::Function<Copies>{
 public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,at::TensorList rows,at::Device device,at::ScalarType dtype){
    ctx->set_materialize_grads(false);ctx->save_for_backward(rows.vec());
    torch::autograd::variable_list flat,outputs;
    for(const auto& t:rows)flat.push_back(t.reshape({-1}));
    auto packed=at::cat(flat).to(rows[0].options().device(device).dtype(dtype));int64_t offset=0;
    for(const auto& t:rows){outputs.push_back(packed.narrow(0,offset,t.numel()).reshape(t.sizes()));offset+=t.numel();}
    torch::autograd::variable_list frozen;
    for(size_t i=0;i<rows.size();++i)if(!rows[i].requires_grad())frozen.push_back(outputs[i]);
    ctx->mark_non_differentiable(frozen);return outputs;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,torch::autograd::variable_list bars){
    if(at::GradMode::is_enabled())throw std::invalid_argument("weighted placement supports first-order VJP only");
    auto rows=ctx->get_saved_variables();torch::autograd::variable_list result(rows.size()+2),used;std::vector<size_t> ids;
    for(size_t i=0;i<bars.size();++i)if(bars[i].defined()&&ctx->needs_input_grad(i)){ids.push_back(i);used.push_back(bars[i]);}
    if(!used.empty()){
      torch::autograd::variable_list flat;for(const auto& t:used)flat.push_back(t.reshape({-1}));
      auto packed=at::cat(flat).to(rows[0].options());int64_t offset=0;
      for(auto i:ids){result[i]=packed.narrow(0,offset,rows[i].numel()).reshape(rows[i].sizes());offset+=rows[i].numel();}
    }
    return result;
  }
};
}
Tensor placement_copy(const Tensor& value,at::Device device,at::ScalarType dtype){
  if(!value.defined())return value;
  if(!value.is_floating_point())dtype=value.scalar_type();
  if(value.device()==device&&value.scalar_type()==dtype)return value;
  if(value.requires_grad()&&at::GradMode::is_enabled())return Copy::apply(value,device,dtype);
  return value.to(value.options().device(device).dtype(dtype));
}
void placement_tensors(const std::vector<Tensor*>& handles,at::Device device,at::ScalarType dtype){
  std::map<const void*,Tensor> unique,mapped;std::map<std::string,std::vector<const void*>> groups;
  // Allocation addresses identify aliases; they must not choose copy/VJP edge
  // order. Restored owners have new addresses but the same ordered fields.
  std::vector<const void*> order;
  for(auto t:handles)if(t->defined()){
    const auto id=t->unsafeGetTensorImpl();
    if(unique.emplace(id,*t).second)order.push_back(id);
  }
  for(const auto id:order){
    const auto& t=unique.at(id);
    auto target=t.is_floating_point()?dtype:t.scalar_type();
    if(t.device()==device&&t.scalar_type()==target){mapped.emplace(id,t);continue;}
    std::ostringstream key;key<<t.device()<<'/'<<int(t.scalar_type())<<'/'<<int(target);groups[key.str()].push_back(id);
  }
  for(const auto& [key,ids]:groups){
    const auto& first=unique.at(ids[0]);auto target=first.is_floating_point()?dtype:first.scalar_type();
    const auto element=std::max<int64_t>(first.element_size(),c10::elementSize(target));
    for(size_t start=0;start<ids.size();){
      size_t end=start;int64_t bytes=0;
      while(end<ids.size()){
        auto size=std::max<int64_t>(1,unique.at(ids[end]).numel())*element;
        if(end>start&&size>8*1024*1024-bytes)break;
        bytes+=size;++end;
      }
      auto count=end-start;std::vector<Tensor> rows;
      for(size_t j=0;j<count;++j)rows.push_back(unique.at(ids[start+j]));
      auto values=count==1?std::vector<Tensor>{placement_copy(rows[0],device,target)}:Copies::apply(at::TensorList(rows),device,target);
      for(size_t j=0;j<count;++j)mapped.emplace(ids[start+j],values[j]);
      start=end;
    }
  }
  std::vector<Tensor> replacements;replacements.reserve(handles.size());
  for(auto t:handles)replacements.push_back(t->defined()?mapped.at(t->unsafeGetTensorImpl()):*t);
  for(size_t i=0;i<handles.size();++i)*handles[i]=replacements[i];
}
}
