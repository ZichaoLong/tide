#include "streaming.h"
#include <algorithm>
#include <sstream>
#include <set>

namespace tide_bench {
const char* difference_name(DifferenceKind kind) {
  switch(kind) {
    case DifferenceKind::numeric: return "numeric";
    case DifferenceKind::discrete: return "discrete";
    case DifferenceKind::structure: return "structure";
    case DifferenceKind::nonfinite: return "nonfinite";
  }
  throw std::invalid_argument("unknown comparison difference kind");
}
ComparisonFailure::ComparisonFailure(Difference value)
  :std::runtime_error("benchmark parity: "+value.path+": "+value.message),difference(std::move(value)) {}
void ComparisonReport::add(Difference value) {
  if(value.kind==DifferenceKind::numeric)++numeric;
  else if(value.kind==DifferenceKind::discrete)++discrete;
  else throw ComparisonFailure(std::move(value));
  if(differences.size()<limit)differences.push_back(std::move(value));
}
void ComparisonReport::merge(const ComparisonReport& other) {
  numeric+=other.numeric;discrete+=other.discrete;
  for(const auto& value:other.differences)if(differences.size()<limit)differences.push_back(value);
}
namespace {
struct Finite {
  std::set<const void*> seen;
  std::map<std::string,std::vector<std::pair<std::string,at::Tensor>>> flags;
  void tensor(const at::Tensor& x,const std::string& path) {
    if(!x.defined()||(!x.is_floating_point()&&!x.is_complex())||!seen.insert(x.unsafeGetTensorImpl()).second)return;
    flags[x.device().str()].push_back({path,at::isfinite(x).all()});
  }
  void slots(const std::map<std::string,at::Tensor>& xs,const std::string& path) {
    for(const auto& [name,x]:xs)tensor(x,path+"."+name);
  }
  void state(const tide::State& s,const std::string& path) {tensor(s.value,path+".value");slots(s.slots,path+".slots");}
  void history(const tide::History& h,const std::string& path) {slots(h.tensors,path+".tensors");}
  void finish() {
    // One host decision per device on the successful path, not per record.
    for(const auto& [device,rows]:flags) {
      std::vector<at::Tensor> values;for(const auto& row:rows)values.push_back(row.second);
      if(at::stack(values).all().item<bool>())continue;
      for(const auto& [path,value]:rows)if(!value.item<bool>())
        throw ComparisonFailure({DifferenceKind::nonfinite,path,"nonfinite tensor"});
    }
  }
};
struct Comparison {
  at::ScalarType payload_dtype;
  double rtol, atol;
  std::optional<at::Device> candidate_device;
  ComparisonReport* report = nullptr;
  bool half_storage_reference = true;
  void require(bool ok,DifferenceKind kind,const std::string& path,const std::string& message) {
    if(ok)return;
    Difference failure{kind,path,message};
    if(report)report->add(std::move(failure));else throw ComparisonFailure(std::move(failure));
  }
  void exact(bool ok,const std::string& path,const std::string& message) {
    require(ok,DifferenceKind::discrete,path,message);
  }
  void tensor(const at::Tensor& actual,const at::Tensor& expected,const std::string& path) {
    exact(actual.defined()==expected.defined(),path,"tensor presence / None-zero mismatch");
    if(!actual.defined()||!expected.defined())return;
    if(candidate_device)require(actual.device()==*candidate_device,DifferenceKind::structure,path,"candidate tensor device");
    auto a=actual.detach().to(at::kCPU),b=expected.detach().to(at::kCPU);
    // The declared FP16 storage reference exposes FP32 arithmetic. Preserve
    // this existing convention; no other dtype conversion is made.
    if(payload_dtype==at::kHalf&&half_storage_reference) {
      if(a.scalar_type()==at::kHalf)a=a.to(at::kFloat);
      if(b.scalar_type()==at::kHalf)b=b.to(at::kFloat);
    }
    require(a.scalar_type()==b.scalar_type()&&a.sizes()==b.sizes()&&a.layout()==b.layout(),
            DifferenceKind::structure,path,"tensor metadata");
    if(!a.is_floating_point()&&!a.is_complex()) {
      exact(at::equal(a,b),path,"integer/bool tensor value");return;
    }
    const bool fp64=a.scalar_type()==at::kDouble&&payload_dtype==at::kDouble;
    if(!at::allclose(a,b,rtol>=0?rtol:fp64?1e-8:1e-5,atol>=0?atol:fp64?1e-10:1e-6)) {
      std::ostringstream message;
      message<<"tensor value (dtype="<<a.scalar_type()<<", shape="<<a.sizes()
             <<", max_abs_error="<<(a-b).abs().max().item<double>()
             <<", reference_max_abs="<<b.abs().max().item<double>()<<')';
      require(false,DifferenceKind::numeric,path,message.str());
    }
  }
  void slots(const std::map<std::string,at::Tensor>& a,const std::map<std::string,at::Tensor>& b,const std::string& path) {
    exact(a.size()==b.size(),path,"tensor slots / owner count");
    for(const auto& [key,value]:a) {
      const auto other=b.find(key);exact(other!=b.end(),path+"."+key,"tensor owner missing");
      if(other!=b.end())tensor(value,other->second,path+"."+key);
    }
  }
  void state(const tide::State& a,const tide::State& b,const std::string& path) {
    exact(a.last_time==b.last_time&&a.observations==b.observations,path,"state clock/count");
    tensor(a.value,b.value,path+".value");slots(a.slots,b.slots,path+".slots");
  }
  void history(const tide::History& a,const tide::History& b,const std::string& path) {
    exact(a.last_time==b.last_time&&a.scalars==b.scalars&&a.node_maps==b.node_maps,path,"history");
    slots(a.tensors,b.tensors,path+".tensors");
  }
  void atom(const tide::Atom& a,const tide::Atom& b,const std::string& path) {
    exact(a.key()==b.key(),path,"atom identity");tensor(a.value,b.value,path+".value");
  }
  void atoms(const std::vector<tide::Atom>& a,const std::vector<tide::Atom>& b,const std::string& path) {
    exact(a.size()==b.size(),path,"atom count");
    for(size_t i=0;i<std::min(a.size(),b.size());++i)atom(a[i],b[i],path+"["+std::to_string(i)+"]");
  }
  void emitted(const std::vector<tide::SlotValue>& a,const std::vector<tide::SlotValue>& b,const std::string& path) {
    exact(a.size()==b.size(),path,"slot count");
    for(size_t i=0;i<std::min(a.size(),b.size());++i) {
      const auto slot=path+"["+std::to_string(i)+"]";
      exact(a[i].slot==b[i].slot,slot,"slot identity");tensor(a[i].value,b[i].value,slot+".value");
    }
  }
  void result(const tide::Result& a,const tide::Result& b,bool traces) {
    const auto& x=a.continuation;const auto& y=b.continuation;
    // Historical dormant-extension anchors have different whole-graph IDs.
    // compare() preserves that domain; collect_compare() additionally checks
    // the same identity for public experiment acceptance.
    exact(!x.identity.empty()&&!y.identity.empty(),"continuation.identity","graph identity presence");
    exact(x.cut==y.cut&&x.batch_size==y.batch_size&&x.ledger==y.ledger,"continuation","cut/batch/ledger");
    exact(x.states.size()==y.states.size()&&x.history.size()==y.history.size(),"continuation","state/history owners");
    for(const auto& [owner,value]:x.states) {
      const auto path="continuation.states["+std::to_string(owner.first)+","+std::to_string(owner.second)+"]";
      const auto other=y.states.find(owner);exact(other!=y.states.end(),path,"state owner missing");
      if(other!=y.states.end())state(value,other->second,path);
    }
    for(const auto& [owner,value]:x.history) {
      const auto path="continuation.history["+std::to_string(owner.first)+","+std::to_string(owner.second)+"]";
      const auto other=y.history.find(owner);exact(other!=y.history.end(),path,"history owner missing");
      if(other!=y.history.end())history(value,other->second,path);
    }
    atoms(x.pending,y.pending,"continuation.pending");exact(a.outputs.size()==b.outputs.size(),"outputs","output count");
    for(size_t i=0;i<std::min(a.outputs.size(),b.outputs.size());++i) {
      const auto& u=a.outputs[i];const auto& v=b.outputs[i];const auto path="outputs["+std::to_string(i)+"]";
      exact(std::tie(u.batch,u.time,u.port)==std::tie(v.batch,v.time,v.port),path,"output identity");
      tensor(u.value,v.value,path+".value");
    }
    if(!traces)return;
    atoms(a.messages,b.messages,"messages");exact(a.trace.size()==b.trace.size(),"trace","trace count");
    for(size_t i=0;i<std::min(a.trace.size(),b.trace.size());++i) {
      const auto& u=a.trace[i];const auto& v=b.trace[i];
      const auto path="trace["+std::to_string(i)+"][b="+std::to_string(u.batch)+",v="+std::to_string(u.node)+",t="+std::to_string(u.time)+"]";
      exact(std::tie(u.batch,u.node,u.time,u.active)==std::tie(v.batch,v.node,v.time,v.active),path,"event/route");
      atoms(u.fiber,v.fiber,path+".fiber");history(u.history,v.history,path+".history");
      state(u.old,v.old,path+".old");state(u.proposed_state,v.proposed_state,path+".proposal");
      state(u.comparison_state,v.comparison_state,path+".comparison");state(u.next_state,v.next_state,path+".next");
      tensor(u.content,v.content,path+".content");tensor(u.proposal,v.proposal,path+".proposal.value");
      tensor(u.descriptor,v.descriptor,path+".descriptor");tensor(u.control,v.control,path+".control");
      tensor(u.comparison,v.comparison,path+".comparison.value");tensor(u.next,v.next,path+".next.value");tensor(u.full,v.full,path+".full");
      slots(u.descriptor_fields,v.descriptor_fields,path+".descriptor_fields");
      emitted(u.emitted,v.emitted,path+".emitted");emitted(u.contributions,v.contributions,path+".contributions");
      exact(u.sources.size()==v.sources.size(),path,"source count");
      for(size_t j=0;j<std::min(u.sources.size(),v.sources.size());++j) {
        const auto source=path+".sources["+std::to_string(j)+"]";
        exact(u.sources[j].slot==v.sources[j].slot,source,"source slot");
        atom(u.sources[j].atom,v.sources[j].atom,source);tensor(u.sources[j].scale,v.sources[j].scale,source+".scale");
      }
    }
  }
};
}  // namespace
void check_finite(const std::map<std::string,at::Tensor>& values,const std::string& path) {
  Finite check;check.slots(values,path);check.finish();
}
void check_finite(const tide::Result& result,bool traces) {
  Finite check;
  const auto& q=result.continuation;
  for(const auto& [owner,s]:q.states)check.state(s,"continuation.states["+std::to_string(owner.first)+","+std::to_string(owner.second)+"]");
  for(const auto& [owner,h]:q.history)check.history(h,"continuation.history["+std::to_string(owner.first)+","+std::to_string(owner.second)+"]");
  for(const auto& a:q.pending)check.tensor(a.value,"continuation.pending");
  for(const auto& o:result.outputs)check.tensor(o.value,"outputs");
  if(!traces){check.finish();return;}
  for(const auto& a:result.messages)check.tensor(a.value,"messages");
  for(size_t i=0;i<result.trace.size();++i) {
    const auto& e=result.trace[i];const auto path="trace["+std::to_string(i)+"]";
    for(const auto* x:{&e.content,&e.proposal,&e.descriptor,&e.control,&e.comparison,&e.next,&e.full})check.tensor(*x,path);
    check.slots(e.descriptor_fields,path+".descriptor_fields");check.slots(e.state_prepared,path+".state_prepared");
    for(const auto* s:{&e.old,&e.proposed_state,&e.comparison_state,&e.next_state})check.state(*s,path);
    check.history(e.history,path+".history");
    for(const auto& a:e.fiber)check.tensor(a.value,path+".fiber");
    for(const auto& x:e.emitted)check.tensor(x.value,path+".emitted");
    for(const auto& x:e.contributions)check.tensor(x.value,path+".contributions");
    for(const auto& s:e.sources){check.tensor(s.atom.value,path+".source");check.tensor(s.scale,path+".scale");}
    if(e.source_batch)check.tensor(e.source_batch->values,path+".source_batch");
  }
  check.finish();
}
void compare(const tide::Result& a,const tide::Result& b,bool traces,at::ScalarType dtype,
             std::optional<at::Device> candidate_device,double rtol,double atol) {
  check_finite(a,traces);check_finite(b,traces);
  Comparison{dtype,rtol,atol,candidate_device}.result(a,b,traces);
}
ComparisonReport collect_compare(const tide::Result& a,const tide::Result& b,bool traces,at::ScalarType dtype,
    std::optional<at::Device> device,double rtol,double atol,size_t maximum) {
  if(!maximum)throw std::invalid_argument("comparison diagnostic limit must be positive");
  check_finite(a,traces);check_finite(b,traces);
  ComparisonReport report;report.limit=maximum;Comparison comparison{dtype,rtol,atol,device,&report};
  comparison.exact(a.continuation.identity==b.continuation.identity,"continuation.identity","different graph identities");
  comparison.result(a,b,traces);return report;
}
ComparisonReport collect_tensors(const std::map<std::string,at::Tensor>& a,const std::map<std::string,at::Tensor>& b,
    at::ScalarType dtype,const std::string& path,double rtol,double atol,size_t maximum) {
  if(!maximum)throw std::invalid_argument("comparison diagnostic limit must be positive");
  check_finite(a,path);check_finite(b,path);ComparisonReport report;report.limit=maximum;
  Comparison{dtype,rtol,atol,{},&report,false}.slots(a,b,path);return report;
}
}  // namespace tide_bench
