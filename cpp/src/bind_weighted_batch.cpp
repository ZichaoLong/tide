// Testing/client adapters; all numerical computation is owned by LibTorch.
#include "tide/weighted.h"
#include "tide/allocation_failure.h"
#include "weighted_sending_batch.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
namespace py=pybind11;
void bind_weighted_batch(py::module_& m){
  using namespace tide;using namespace tide::weighted;
  py::register_exception_translator([](std::exception_ptr pointer){
    try{if(pointer)std::rethrow_exception(pointer);}
    catch(const AllocationFailure& e){
      auto type=py::module_::import("tidegraph.weighted.allocation_failure").attr("AllocationFailure");
      auto value=type(e.what(),e.row,e.tensors,e.details);
      value.attr("time")=py::cast(e.time);value.attr("nodes")=py::cast(e.nodes);value.attr("members")=py::cast(e.members);
      PyErr_SetObject(type.ptr(),value.ptr());
    }
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
