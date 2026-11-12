// Tensor-preserving client adapter; C++ owns the whole schedule and local math.
#include "tide/weighted_sequence_columns.h"
#include "weighted_row_linear.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
namespace py=pybind11;
void bind_weighted_sequence_columns(py::module_& m){
  using namespace tide;using namespace tide::weighted::columns;
  using Result=tide::weighted::columns::Result;
  m.def("weighted_linear_blocks",&tide::weighted::linear_blocks);
  py::class_<Layout>(m,"ColumnLayout").def(py::init<std::vector<Index>,std::vector<Index>>())
    .def_readonly("offsets",&Layout::offsets).def_readonly("times",&Layout::times);
  py::class_<KVState>(m,"KVColumnState").def(py::init<Layout,TensorFields,Tensor,std::vector<Index>,std::vector<Index>>())
    .def_readonly("layout",&KVState::layout).def_readonly("fields",&KVState::fields)
    .def_readonly("inert",&KVState::inert).def_readonly("last_times",&KVState::last_times)
    .def_readonly("observations",&KVState::observations);
  py::class_<KVWindow>(m,"KVColumnWindow")
    .def_readonly("layout",&KVWindow::layout).def_readonly("bank_layout",&KVWindow::bank_layout)
    .def_readonly("mass",&KVWindow::mass).def_readonly("content",&KVWindow::content)
    .def_readonly("bank",&KVWindow::bank).def_readonly("terminal",&KVWindow::terminal);
  py::class_<RecurrentState>(m,"RecurrentColumnState")
    .def(py::init([](TensorFields fields,Tensor inert,std::vector<Index> times,std::vector<Index> counts,
                    Layout cache,std::optional<Tensor> values){
      return RecurrentState{std::move(fields),inert,std::move(times),std::move(counts),std::move(cache),values.value_or(Tensor{})};}))
    .def_readonly("fields",&RecurrentState::fields).def_readonly("inert",&RecurrentState::inert)
    .def_readonly("last_times",&RecurrentState::last_times).def_readonly("observations",&RecurrentState::observations)
    .def_readonly("cache_layout",&RecurrentState::cache_layout).def_readonly("cache_values",&RecurrentState::cache_values);
  py::class_<RecurrentWindow>(m,"RecurrentColumnWindow")
    .def_readonly("layout",&RecurrentWindow::layout).def_readonly("write_layout",&RecurrentWindow::write_layout)
    .def_readonly("mass",&RecurrentWindow::mass).def_readonly("content",&RecurrentWindow::content)
    .def_readonly("bank",&RecurrentWindow::bank).def_readonly("times",&RecurrentWindow::times)
    .def_readonly("cache_layout",&RecurrentWindow::cache_layout).def_readonly("cache_values",&RecurrentWindow::cache_values)
    .def_readonly("terminal",&RecurrentWindow::terminal);
  py::class_<Message>(m,"MessageColumns").def(py::init<std::vector<Index>,Tensor,Tensor>())
    .def_readonly("rows",&Message::rows).def_readonly("strength",&Message::strength).def_readonly("content",&Message::content);
  py::class_<NodeResult>(m,"NodeColumns").def_readonly("rows",&NodeResult::rows)
    .def_readonly("window",&NodeResult::window).def_readonly("descriptors",&NodeResult::descriptors).def_readonly("controls",&NodeResult::controls);
  py::class_<Result>(m,"SettleColumns").def_readonly("layout",&Result::layout)
    .def_readonly("states",&Result::states).def_readonly("outputs",&Result::outputs).def_readonly("messages",&Result::messages)
    .def_readonly("nodes",&Result::nodes).def_readonly("stats",&Result::stats).def("readout",&readout);
  m.def("weighted_settle_columns",&settle,py::arg("spec"),py::arg("model"),py::arg("layout"),py::arg("ports"),
        py::arg("left"),py::arg("max_workspace_bytes"),py::arg("placement")=std::nullopt,py::call_guard<py::gil_scoped_release>());
  m.attr("weighted_column_placement_version")=3;
}
