// Tensor-preserving client adapter; C++ owns the whole schedule and local math.
#include "tide/weighted_sequence_columns.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
namespace py=pybind11;
void bind_weighted_sequence_columns(py::module_& m){
  using namespace tide;using namespace tide::weighted::columns;
  using Result=tide::weighted::columns::Result;
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
  py::class_<Message>(m,"MessageColumns").def(py::init<std::vector<Index>,Tensor,Tensor>())
    .def_readonly("rows",&Message::rows).def_readonly("strength",&Message::strength).def_readonly("content",&Message::content);
  py::class_<NodeResult>(m,"NodeColumns").def_readonly("rows",&NodeResult::rows)
    .def_readonly("window",&NodeResult::window).def_readonly("descriptors",&NodeResult::descriptors).def_readonly("controls",&NodeResult::controls);
  py::class_<Result>(m,"SettleColumns").def_readonly("layout",&Result::layout)
    .def_readonly("states",&Result::states).def_readonly("outputs",&Result::outputs).def_readonly("messages",&Result::messages)
    .def_readonly("nodes",&Result::nodes).def_readonly("stats",&Result::stats).def("readout",&readout);
  m.def("weighted_settle_columns",&settle,py::call_guard<py::gil_scoped_release>());
}
