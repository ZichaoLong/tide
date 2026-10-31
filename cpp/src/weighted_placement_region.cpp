// Control/query and allocation locations are independent of payload owners.
#include "weighted_placement.h"
#include "weighted_region_batch.h"
#include "region_metadata.h"
namespace tide::weighted {
namespace {
struct Mapped {
  RegionWeights weights;
  std::vector<std::vector<Candidate>> candidates;
  std::vector<RegionInput> requests;
  Mapped(const RegionWeights& w,const std::vector<RegionInput>& rows,at::Device device)
      :weights(w),candidates(rows.size()){
    auto dtype=rows[0].candidates[0].fields.at("w").scalar_type();
    std::vector<Tensor*> handles;
    for(auto& [name,t]:weights.extra)handles.push_back(&t);
    for(size_t i=0;i<rows.size();++i){
      const auto& r=rows[i];candidates[i]={r.candidates.begin(),r.candidates.end()};
      for(auto& d:candidates[i])for(auto& [name,t]:d.fields)handles.push_back(&t);
    }
    placement_tensors(handles,device,dtype);
    for(size_t i=0;i<rows.size();++i){const auto& r=rows[i];requests.push_back({r.history,r.time,candidates[i],r.layout,r.payload_options});}
  }
};
Selection finish(Selection result,const RegionInput& r){
  std::set<Index> positive;
  for(auto& [node,c]:result.controls){
    c=placement_copy(c,r.payload_options.device(),r.payload_options.dtype().toScalarType());
    if(c.detach().item<double>()>0)positive.insert(node);
  }
  if(positive!=result.active)throw std::invalid_argument("weighted control cast changes positive support");
  return result;
}
std::vector<Selection> finish_batch(std::vector<Selection> results,const std::vector<RegionInput>& requests){
  bool cpu=true;for(const auto& r:requests)cpu=cpu&&r.payload_options.device().is_cpu();
  if(results.size()<2||cpu){
    for(size_t i=0;i<results.size();++i)results[i]=finish(std::move(results[i]),requests[i]);
    return results;
  }
  std::vector<Tensor> controls;
  for(size_t i=0;i<results.size();++i){const auto& r=requests[i];
    for(auto& [node,c]:results[i].controls){
      c=placement_copy(c,r.payload_options.device(),r.payload_options.dtype().toScalarType());controls.push_back(c);
    }
  }
  const auto support=region_metadata::flags(controls,true);size_t offset=0;
  for(const auto& result:results){std::set<Index> positive;
    for(const auto& [node,c]:result.controls)if(support[offset++])positive.insert(node);
    if(positive!=result.active)throw std::invalid_argument("weighted control cast changes positive support");
  }
  return results;
}
class PlacedSending final:public RegionKernel {
 public:
  PlacedSending(std::shared_ptr<const RegionKernel> original,RegionalConfig c,ResolvedPlacement p)
      :original_(std::move(original)),config_(std::move(c)),placement_(std::move(p)){}
  bool joint_batch()const override{return true;}
  bool batched_autograd()const override{return true;}
  bool history_independent()const override{return true;}
  History initial(const RegionWeights& w,const RegionLayout& layout,const Tensor& ref)const override{return original_->initial(w,layout,ref);}
  void validate_weights(const RegionWeights& w,const RegionLayout& layout)const override{original_->validate_weights(w,layout);}
  void validate_history(const History& h,const RegionLayout& layout)const override{original_->validate_history(h,layout);}
  Selection step(const RegionWeights& w,const RegionInput& r)const override{
    Mapped mapped(w,{r},placement_.control);
    return finish(region_step(mapped.weights,mapped.requests[0],config_,placement_.selection),r);
  }
  std::vector<Selection> batch(const RegionWeights& w,const std::vector<RegionInput>& r)const override{return evaluate(w,r,false);}
  std::vector<Selection> batch_grad(const RegionWeights& w,const std::vector<RegionInput>& r)const override{return evaluate(w,r,true);}
  std::shared_ptr<const RegionKernel> original()const{return original_;}
 private:
  std::vector<Selection> evaluate(const RegionWeights& w,const std::vector<RegionInput>& rows,bool grad)const{
    std::map<at::ScalarType,std::vector<size_t>> groups;std::vector<Selection> result(rows.size());
    for(size_t i=0;i<rows.size();++i)groups[rows[i].candidates[0].fields.at("w").scalar_type()].push_back(i);
    for(const auto& [dtype,ids]:groups){
      std::vector<RegionInput> requests;for(auto i:ids)requests.push_back(rows[i]);
      Mapped mapped(w,requests,placement_.control);
      auto values=grad?batch::region_vjp(mapped.weights,mapped.requests,config_,placement_.selection)
        :batch::regions(mapped.weights,mapped.requests,config_,nullptr,placement_.selection);
      values=finish_batch(std::move(values),requests);
      for(size_t j=0;j<ids.size();++j)result[ids[j]]=std::move(values[j]);
    }
    return result;
  }
  std::shared_ptr<const RegionKernel> original_;RegionalConfig config_;ResolvedPlacement placement_;
};
}
std::shared_ptr<const RegionKernel> place_region(const RegionWeights& w,const ResolvedPlacement& p){
  auto original=w.kernel;if(!original)return {};
  if(auto prior=dynamic_cast<const PlacedSending*>(original.get()))original=prior->original();
  auto config=sending_config(*original);if(!config)return {};
  return std::make_shared<PlacedSending>(original,*config,p);
}
}
