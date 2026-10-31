#include "roots.h"
#include "eager.h"
#include "validation.h"
#include <ATen/core/grad_mode.h>
#include <set>
namespace weighted_benchmark {
namespace {
constexpr int64_t parameter_limit=5000000;
Json gradient_inventory(const std::map<std::string,Tensor>& values){
  size_t connected=0,zero=0;for(const auto& [_,value]:values)if(value.defined()){
    ++connected;if(!value.detach().to(at::kCPU).count_nonzero().item<int64_t>())++zero;
  }
  return {{"leaves",values.size()},{"connected",connected},{"connected_zero",zero},{"disconnected",values.size()-connected}};
}
void merge(tide_bench::ComparisonReport& into,tide_bench::ComparisonReport part,const std::string& prefix){
  for(auto& row:part.differences)row.path=prefix+"."+row.path;into.merge(part);
}
}
void validate_root_budget(const Json& data){
  // Count unique trainable parameter table entries before opening binary data
  // or constructing either model. Aliases remain one physical owner.
  std::set<size_t> indices;
  for(const auto& node:data.at("nodes")){
    for(const auto* key:{"decay","weight","bias","read"})indices.insert(node.at(key).get<size_t>());
    for(const auto& [_,index]:node.at("extra").items())indices.insert(index.get<size_t>());
  }
  for(const auto& region:data.at("region_weights"))for(const auto& [_,index]:region.items())indices.insert(index.get<size_t>());
  for(const auto& [_,values]:data.at("scales").items())for(const auto& index:values)indices.insert(index.get<size_t>());
  int64_t total=0;
  for(auto index:indices){const auto& row=data.at("tensors").at(index);if(!row.at("trainable").get<bool>())continue;
    int64_t count=1;for(auto extent:row.at("shape").get<std::vector<int64_t>>()){
      if(extent<0||extent>parameter_limit||(extent&&count>parameter_limit/extent))
        throw std::invalid_argument("--check-roots requires <= 5,000,000 unique trainable parameters; use a small representative fixture");
      count*=extent;
    }
    if(count>parameter_limit-total)throw std::invalid_argument("--check-roots requires <= 5,000,000 unique trainable parameters; use a small representative fixture");
    total+=count;
  }
}
Json check_roots(const Json& data,const Request& request,at::Device target,const ExecutionPlacement& placement,
                 const std::vector<at::Device>& peers,const std::vector<at::Device>& nodes,
                 tide_bench::ComparisonReport& total){
  validate_root_budget(data);at::AutoGradMode grad(true);const auto dtype=request.dtype();
  const double rtol=dtype==at::kDouble?1e-8:dtype==at::kHalf?2e-2:1e-5;
  const double atol=dtype==at::kDouble?1e-10:dtype==at::kHalf?1e-3:1e-6;
  ExecutionPlacement cpu;cpu.scoring_dtype=request.scoring;
  Eager candidate(data,false,request.schedule=="prefill",request.packing=="scalar",dtype,target,
                  placement,request.workers,peers,false,false,nodes);
  Eager reference(data,false,request.schedule!="prefill",true,dtype,at::Device(at::kCPU),cpu);
  tide_bench::ComparisonReport report;Json cases=Json::array();
  for(bool split:{false,true}){
    const std::string scope=split?"split-continuation":"whole-first-window";
    const auto actual=candidate.diagnostic(split),expected=reference.diagnostic(split);
    merge(report,tide_bench::collect_compare(Result{actual.initial},Result{expected.initial},false,dtype,{},rtol,atol),scope+".initial");
    merge(report,tide_bench::collect_compare(actual.result,expected.result,true,dtype,{},rtol,atol),scope+".forward");
    const auto a=diagnostic_roots(actual),b=diagnostic_roots(expected);
    report.merge(tide_bench::collect_tensors(a,b,dtype,scope+".objectives",rtol,atol));
    for(const auto& [name,root]:a)for(bool zero:{false,true}){
      const auto path=scope+"."+name+(zero?".zero":".nonzero");
      tide_bench::ComparisonReport part;
      if(root.requires_grad()!=b.at(name).requires_grad())
        part.add({tide_bench::DifferenceKind::discrete,path,"root autograd connectivity differs"});
      const auto av=diagnostic_vjp(root,actual.leaves,zero),bv=diagnostic_vjp(b.at(name),expected.leaves,zero);
      part.merge(tide_bench::collect_tensors(av,bv,dtype,path,rtol,atol));
      cases.push_back({{"scope",scope},{"start",actual.start},{"stop",actual.stop},{"root",name},
        {"cotangent",zero?0.:.7},{"candidate",gradient_inventory(av)},{"reference",gradient_inventory(bv)},
        {"comparison",comparison_json(part)}});report.merge(part);
    }
  }
  total.merge(report);auto record=comparison_json(report);
  record["schema"]="tide-weighted-independent-roots-v1";record["state"]=report.passed()?"passed":"failed";
  record["parameter_limit"]=parameter_limit;record["cases"]=cases;
  record["scope"]="standalone eager selected device/placement/schedule against independent scalar CPU opposite schedule; original fixture, no optimizer update";
  record["objective"]="sum of per-field mean squares; FP16 fields promoted to FP32; distinct all/output/control/descriptor objectives";
  record["leaves"]="canonical shared parameters, canonical external aliases; split independently cloned floating state/slots/history/physical pending";
  return record;
}
}  // namespace weighted_benchmark
