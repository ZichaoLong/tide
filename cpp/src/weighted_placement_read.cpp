// Only the pinned Describe is adapted; general custom Read remains explicit.
#include "weighted_placement.h"
namespace tide::weighted {
namespace {
struct Mapped {
  NodeWeights weights;
  std::vector<State> states;
  std::vector<ReadInput> requests;
  Mapped(const NodeWeights& original,const std::vector<ReadInput>& rows,at::Device device,at::ScalarType dtype)
      :weights(original),states(rows.size()){
    std::vector<Tensor*> handles{&weights.decay,&weights.weight,&weights.bias,&weights.read};
    for(auto& [name,t]:weights.extra)handles.push_back(&t);
    std::vector<ContentView> contents;contents.reserve(rows.size());
    for(size_t i=0;i<rows.size();++i){
      const auto& r=rows[i];contents.push_back(r.content);handles.push_back(&contents.back().value);
      if(r.state){states[i]=*r.state;handles.push_back(&states[i].value);for(auto& [name,t]:states[i].slots)handles.push_back(&t);}
    }
    placement_tensors(handles,device,dtype);
    for(size_t i=0;i<rows.size();++i)requests.push_back({rows[i].state?&states[i]:nullptr,rows[i].time,contents[i]});
  }
};
class PlacedDescribe final:public ReadKernel {
 public:
  PlacedDescribe(std::shared_ptr<const ReadKernel> original,at::Device device,at::ScalarType dtype)
      :original_(std::move(original)),device_(device),dtype_(dtype){}
  bool structured()const override{return true;}
  bool joint_batch()const override{return true;}
  bool batched_autograd()const override{return true;}
  FieldLayout field_layout(const NodeWeights& w)const override{return original_->field_layout(w);}
  at::ScalarType descriptor_dtype(at::ScalarType)const override{return dtype_;}
  at::Device descriptor_device(at::Device)const override{return device_;}
  void validate_weights(const NodeWeights& w)const override{original_->validate_weights(w);}
  TensorFields fields_step(const NodeWeights& w,const ReadInput& r)const override{
    Mapped mapped(w,{r},device_,dtype_);return original_->fields_step(mapped.weights,mapped.requests[0]);
  }
  std::vector<TensorFields> fields_batch(const NodeWeights& w,const std::vector<ReadInput>& r)const override{
    Mapped mapped(w,r,device_,dtype_);return original_->fields_batch(mapped.weights,mapped.requests);
  }
  std::vector<TensorFields> fields_batch_grad(const NodeWeights& w,const std::vector<ReadInput>& r)const override{
    Mapped mapped(w,r,device_,dtype_);return original_->fields_batch_grad(mapped.weights,mapped.requests);
  }
  std::shared_ptr<const ReadKernel> original()const{return original_;}
 private:
  std::shared_ptr<const ReadKernel> original_;at::Device device_;at::ScalarType dtype_;
};
}
std::shared_ptr<const ReadKernel> place_read(const NodeWeights& w,const ResolvedPlacement& p){
  auto original=w.read_kernel;if(!original)return {};
  if(auto prior=dynamic_cast<const PlacedDescribe*>(original.get()))original=prior->original();
  if(!is_describe(*original))return {};
  auto dtype=w.bias.scalar_type();
  if(p.scoring_dtype=="float32")dtype=at::kFloat;
  if(p.scoring_dtype=="float64")dtype=at::kDouble;
  if(dtype==at::kDouble&&(p.read.type()==c10::DeviceType::PrivateUse1||p.control.type()==c10::DeviceType::PrivateUse1||p.selection.type()==c10::DeviceType::PrivateUse1))
    throw std::invalid_argument("NPU weighted Read/control/selection cannot consume FP64 descriptors");
  return std::make_shared<PlacedDescribe>(original,p.read,dtype);
}
}
