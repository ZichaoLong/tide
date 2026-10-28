#include "streaming.h"
#include <sstream>
#include <stdexcept>

namespace tide_bench {
namespace {
void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(std::string("benchmark parity: ")+what);
}
struct Comparison {
  at::ScalarType payload_dtype;
  double rtol, atol;
  std::optional<at::Device> candidate_device;
void tensor(const at::Tensor& actual, const at::Tensor& expected, const std::string& path="tensor") {
  require(actual.defined() == expected.defined(), "tensor presence");
  if (!actual.defined()) return;
  if (candidate_device) require(actual.device() == *candidate_device, "candidate tensor device");
  auto a = actual.detach().to(at::kCPU), b = expected.detach().to(at::kCPU);
  if(payload_dtype==at::kHalf) {
    if(a.scalar_type()==at::kHalf)a=a.to(at::kFloat);
    if(b.scalar_type()==at::kHalf)b=b.to(at::kFloat);
  }
  require(at::isfinite(a).all().item<bool>() && at::isfinite(b).all().item<bool>(), "nonfinite tensor");
  require(a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes(), "tensor metadata");
  if(!a.is_floating_point()&&!a.is_complex()){
    require(at::equal(a,b),"integer/bool tensor value");return;
  }
  // FP64 descriptors computed from FP32 intermediates inherit FP32 error.
  const bool fp64 = a.scalar_type() == at::kDouble && payload_dtype==at::kDouble;
  if (!at::allclose(a, b, rtol>=0 ? rtol : fp64 ? 1e-8 : 1e-5, atol>=0 ? atol : fp64 ? 1e-10 : 1e-6)) {
    std::ostringstream message;
    message << "benchmark parity: " << path << " tensor value (dtype=" << a.scalar_type() << ", shape=" << a.sizes()
            << ", max_abs_error=" << (a-b).abs().max().item<double>()
            << ", reference_max_abs=" << b.abs().max().item<double>() << ')';
    throw std::runtime_error(message.str());
  }
}
void slots(const std::map<std::string, at::Tensor>& a, const std::map<std::string, at::Tensor>& b,const std::string& path) {
  require(a.size() == b.size(), "tensor slots");
  for (const auto& [key, value] : a) tensor(value, b.at(key),path+"."+key);
}
void state(const tide::State& a, const tide::State& b,const std::string& path) {
  require(a.last_time == b.last_time && a.observations == b.observations, "state clock/count");
  tensor(a.value, b.value,path+".value"); slots(a.slots, b.slots,path+".slots");
}
void history(const tide::History& a, const tide::History& b,const std::string& path) {
  require(a.last_time == b.last_time && a.scalars == b.scalars && a.node_maps == b.node_maps, "history");
  slots(a.tensors, b.tensors,path+".tensors");
}
void atom(const tide::Atom& a, const tide::Atom& b,const std::string& path) {
  require(a.key() == b.key(), "atom identity"); tensor(a.value, b.value,path+".value");
}
void atoms(const std::vector<tide::Atom>& a, const std::vector<tide::Atom>& b,const std::string& path) {
  require(a.size() == b.size(), "atom count");
  for (size_t i = 0; i < a.size(); ++i) atom(a[i], b[i],path+"["+std::to_string(i)+"]");
}
void emitted(const std::vector<tide::SlotValue>& a, const std::vector<tide::SlotValue>& b,const std::string& path) {
  require(a.size() == b.size(), "slot count");
  for (size_t i = 0; i < a.size(); ++i) {
    require(a[i].slot == b[i].slot, "slot identity"); tensor(a[i].value, b[i].value,path+"["+std::to_string(i)+"]");
  }
}
void result(const tide::Result& a, const tide::Result& b, bool traces) {
  const auto& x = a.continuation; const auto& y = b.continuation;
  // The anchor contains only the active rings. All their IDs and edges form an
  // unchanged prefix of the dormant extension; only whole-graph identity differs.
  require(!x.identity.empty() && !y.identity.empty(), "graph identity presence");
  require(x.cut == y.cut && x.batch_size == y.batch_size && x.ledger == y.ledger, "cut/batch/ledger");
  require(x.states.size() == y.states.size() && x.history.size() == y.history.size(), "state/history owners");
  for (const auto& [owner, value] : x.states) state(value, y.states.at(owner),"continuation.states["+std::to_string(owner.first)+","+std::to_string(owner.second)+"]");
  for (const auto& [owner, value] : x.history) history(value, y.history.at(owner),"continuation.history["+std::to_string(owner.first)+","+std::to_string(owner.second)+"]");
  atoms(x.pending, y.pending,"continuation.pending");
  require(a.outputs.size() == b.outputs.size(), "output count");
  for (size_t i = 0; i < a.outputs.size(); ++i) {
    const auto& u = a.outputs[i]; const auto& v = b.outputs[i];
    require(std::tie(u.batch, u.time, u.port) == std::tie(v.batch, v.time, v.port), "output identity");
    tensor(u.value, v.value,"outputs["+std::to_string(i)+"].value");
  }
  if (!traces) return;
  atoms(a.messages, b.messages,"messages"); require(a.trace.size() == b.trace.size(), "trace count");
  for (size_t i = 0; i < a.trace.size(); ++i) {
    const auto& u = a.trace[i]; const auto& v = b.trace[i];
    const auto path="trace["+std::to_string(i)+"][b="+std::to_string(u.batch)+",v="+std::to_string(u.node)+",t="+std::to_string(u.time)+"]";
    require(std::tie(u.batch, u.node, u.time, u.active) == std::tie(v.batch, v.node, v.time, v.active), "event/route");
    atoms(u.fiber, v.fiber,path+".fiber"); history(u.history, v.history,path+".history");
    state(u.old, v.old,path+".old"); state(u.proposed_state, v.proposed_state,path+".proposal");
    state(u.comparison_state, v.comparison_state,path+".comparison"); state(u.next_state, v.next_state,path+".next");
    tensor(u.content, v.content,path+".content"); tensor(u.proposal, v.proposal,path+".proposal.value"); tensor(u.descriptor, v.descriptor,path+".descriptor");
    tensor(u.control, v.control,path+".control"); tensor(u.comparison, v.comparison,path+".comparison.value"); tensor(u.next, v.next,path+".next.value"); tensor(u.full, v.full,path+".full");
    slots(u.descriptor_fields,v.descriptor_fields,path+".descriptor_fields");
    emitted(u.emitted, v.emitted,path+".emitted"); emitted(u.contributions, v.contributions,path+".contributions");
    require(u.sources.size() == v.sources.size(), "source count");
    for (size_t j = 0; j < u.sources.size(); ++j) {
      require(u.sources[j].slot == v.sources[j].slot, "source slot");
      atom(u.sources[j].atom, v.sources[j].atom,path+".sources["+std::to_string(j)+"]"); tensor(u.sources[j].scale, v.sources[j].scale,path+".source.scale");
    }
  }
}
};
}  // namespace
void compare(const tide::Result& a, const tide::Result& b, bool traces, at::ScalarType payload_dtype,
             std::optional<at::Device> candidate_device, double rtol, double atol) {
  Comparison{payload_dtype, rtol, atol, candidate_device}.result(a, b, traces);
}
}  // namespace tide_bench
