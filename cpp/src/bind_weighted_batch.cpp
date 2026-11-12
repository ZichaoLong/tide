// Testing/client adapters; all numerical computation is owned by LibTorch.
#include "tide/weighted.h"
#include "tide/weighted_columns.h"
#include "tide/allocation_failure.h"
#include "tide/allocation_observation.h"
#include "weighted_sending_batch.h"
#include "weighted_regional_groups.h"
#include "weighted_unit_transport.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
namespace py=pybind11;
void bind_weighted_batch(py::module_& m){
  using namespace tide;using namespace tide::weighted;
  m.def("weighted_validate_unit_scales",&validate_unit_scales,
        py::arg("scales"),py::arg("budget")=8*1024*1024,py::arg("fixed")=true,
        py::call_guard<py::gil_scoped_release>());
  py::register_exception_translator([](std::exception_ptr pointer){
    try{if(pointer)std::rethrow_exception(pointer);}
    catch(const AllocationFailure& e){
      auto type=py::module_::import("tidegraph.weighted.allocation_failure").attr("AllocationFailure");
      auto value=type(e.what(),e.row,e.tensors,e.details);
      value.attr("time")=py::cast(e.time);value.attr("nodes")=py::cast(e.nodes);value.attr("members")=py::cast(e.members);
      PyErr_SetObject(type.ptr(),value.ptr());
    }
  });
  m.def("begin_allocation_observation",&begin_allocation_observation);
  m.def("end_allocation_observation",[](){
    auto out=end_allocation_observation();return std::make_tuple(out.rows,out.dropped,out.bytes);
  });
  using Row=std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor,std::optional<Tensor>>;
  m.def("weighted_allocation_batch",[](const std::vector<Row>& rows,const std::string& regularizer,std::optional<double> alpha){
    std::vector<AllocationRow> inputs;
    for(const auto& [s,c,l,u,t,shift]:rows)inputs.push_back({s,c,l,u,t,shift.value_or(Tensor{})});
    std::vector<std::tuple<Tensor,Tensor,std::string,Tensor,Tensor,std::string>> result;
    for(const auto& a:allocate_batch(inputs,regularizer,alpha))result.emplace_back(a.controls,a.multiplier,a.constraint,a.free,a.residual,a.backward);
    return result;
  },py::call_guard<py::gil_scoped_release>());
  m.def("weighted_attention_batch",&batch::attention,py::call_guard<py::gil_scoped_release>());
  m.def("weighted_allocation_columns",[](const Tensor& scores,const Tensor& caps,const Tensor& lower,
      const Tensor& upper,const Tensor& temperature,const std::string& kind,std::optional<double> alpha,
      std::optional<Tensor> shift){
    auto a=allocate_columns(scores,caps,lower,upper,temperature,kind,alpha,shift.value_or(Tensor{}));
    return std::make_tuple(a.controls,a.multiplier,a.branch,a.response,a.free,a.residual,a.backward);
  },py::call_guard<py::gil_scoped_release>());
  m.def("weighted_region_columns",[](const RegionWeights& w,const TensorFields& fields,std::optional<Tensor> nodes){
    return region_columns(w,fields,region_binding(w),nodes.value_or(Tensor{})).controls;
  },py::call_guard<py::gil_scoped_release>());
  m.def("weighted_region_groups",[](const RegionWeights& w,const std::map<Index,std::vector<Index>>& rows,
      const columns::RegionalFields& fields,std::optional<ExecutionPlacement> request){
    columns::RegionalPacking packing(rows);
    if(packing.groups.empty())return std::map<Index,Tensor>{};
    auto placement=resolve_placement(request.value_or(ExecutionPlacement{}),fields.begin()->second.at("w").device());
    return columns::region_groups(w,region_binding(w),packing,fields,placement);
  },py::arg("weights"),py::arg("rows"),py::arg("fields"),py::arg("placement")=std::nullopt,
     py::call_guard<py::gil_scoped_release>());
  m.def("weighted_regional_packing",[](const std::map<Index,std::vector<Index>>& rows,const columns::RegionalFields& fields){
    columns::RegionalPacking packing(rows);std::vector<std::vector<Index>> domains;
    std::vector<Tensor> nodes,controls;std::vector<TensorFields> blocks;
    for(const auto& group:packing.groups){auto p=group.pack(fields);domains.push_back(group.rows);
      nodes.push_back(p.nodes);controls.push_back(p.fields.at("w"));blocks.push_back(std::move(p.fields));}
    auto unpacked=packing.unpack(controls,fields.empty()?Tensor{}:fields.begin()->second.at("w"));
    return std::make_tuple(domains,nodes,blocks,unpacked);
  },py::call_guard<py::gil_scoped_release>());
  m.def("weighted_proportional_batch",&batch::proportional,py::call_guard<py::gil_scoped_release>());
  m.def("weighted_sorted_batch",&batch::sorted_send,py::call_guard<py::gil_scoped_release>());
  m.def("weighted_region_probe",[](const RegionWeights& w,const Region& spec,const std::vector<Index>& members,
      const std::vector<std::vector<std::pair<Index,TensorFields>>>& fields,const std::vector<Index>& times,bool grad){
    if(fields.size()!=times.size())throw std::invalid_argument("region row domain mismatch");
    std::vector<std::vector<Candidate>> candidates;History history;
    for(const auto& row:fields){
      if(row.empty())throw std::invalid_argument("region row must be nonempty");
      candidates.emplace_back();for(const auto& [node,d]:row)candidates.back().push_back({node,{},d});
    }
    std::vector<RegionInput> requests;
    for(size_t i=0;i<times.size();++i)requests.push_back({history,times[i],candidates[i],{spec,members},candidates[i][0].fields.at("w").options()});
    auto values=grad?w.kernel->batch_grad(w,requests):w.kernel->batch(w,requests);
    std::vector<std::tuple<std::set<Index>,std::map<Index,Tensor>,Index>> result;
    for(const auto& value:values)result.emplace_back(value.active,value.controls,value.history.last_time);
    return result;
  },py::call_guard<py::gil_scoped_release>());
}
