#include "weighted_window_check.h"
namespace tide::device_online::window_check {
const at::Tensor& field(const State& s,const std::string& name){return name=="value"?s.value:s.slots.at(name);}
int64_t stage(const Records& data,int64_t v,int64_t b,int64_t time){
  const auto prefix="node."+std::to_string(v)+".";const auto& present=data.at(prefix+"present");const auto& times=data.at(prefix+"times");int64_t out=-1;
  for(int64_t t=0;t<present.size(0);++t)if(present[t][b].item<bool>()&&times[t][b].item<int64_t>()==time){
    if(out>=0)throw std::runtime_error("duplicate actual journal event");out=t;}
  if(out<0)throw std::runtime_error("missing actual journal event");return out;
}
namespace {
void clear(WeightedRowAdjoint r){r.value.fill_(NAN);r.connected.zero_();}
void clear(WeightedRawAdjoint r){clear(WeightedRowAdjoint{r.value,r.connected});}
at::Tensor values(at::IntArrayRef shape){int64_t n=1;for(auto x:shape)n*=x;
  return (.2*at::cos(at::arange(n,at::kFloat)*.17+.3)).reshape(shape);}
void fill_root(const at::Tensor& x,const at::Tensor& connected){x.copy_(values(x.sizes()));connected.fill_(true);}
void fill_root(const WeightedRawLayout& l,const WeightedRawAdjoint& r,const State& s){
  r.value.zero_();for(size_t j=0;j<l.fields().size();++j){const auto& part=l.fields()[j];const auto& x=field(s,part.name);
    r.value.narrow(0,part.offset,x.numel()).copy_(values(x.sizes()).reshape({-1}));r.connected[j]=true;}
}
}
void seed(WeightedWindowAdjoint& vjp,const Records& data,const Result& result,const std::string& mode){
  auto& roots=vjp.roots();for(auto& n:roots.nodes){clear(n.content);clear(n.control);clear(n.full);clear(n.old);clear(n.proposal);clear(n.final);
    for(auto& [name,r]:n.describe)clear(r);}
  clear(roots.pending);clear(roots.messages);clear(roots.fibers);if(mode=="none")return;
  auto choose=[&](const std::string& field){return mode==field||mode=="all";};
  for(const auto& e:result.trace){const auto t=stage(data,e.node,e.batch,e.time),b=e.batch;auto& r=roots.nodes[e.node];
    if(choose("content"))fill_root(r.content.value[t][b],r.content.connected[t][b]);
    if(choose("control"))fill_root(r.control.value[t][b],r.control.connected[t][b]);
    if(e.active&&choose("full")){fill_root(r.full.value[t][b],r.full.connected[t][b]);fill_root(r.control.value[t][b],r.control.connected[t][b]);}
    if(choose("describe"))for(auto& [name,g]:r.describe)fill_root(g.value[t][b],g.connected[t][b]);
    const auto& layout=vjp.nodes()[e.node]->layout;
    if(choose("old"))fill_root(layout,{r.old.value[t][b],r.old.connected[t][b]},e.old);
    if(choose("proposal"))fill_root(layout,{r.proposal.value[t][b],r.proposal.connected[t][b]},e.proposed_state);
  }
  if(choose("final"))for(const auto& [key,s]:result.continuation.states){auto& r=roots.nodes[key.second].final;
    fill_root(vjp.nodes()[key.second]->layout,{r.value[key.first],r.connected[key.first]},s);}
  for(const std::string name:{"messages","fibers"})if(choose(name)){
    auto& r=name=="messages"?roots.messages:roots.fibers;const auto& present=data.at(name=="messages"?"message.present":"fiber.present");
    for(int64_t t=0;t<present.size(0);++t)for(int64_t i=0;i<present.size(1);++i)if(present[t][i].item<bool>())fill_root(r.value[t][i],r.connected[t][i]);}
  if(choose("pending")){const auto& present=data.at("right.pending.present");for(int64_t i=0;i<present.numel();++i)
    if(present[i].item<bool>())fill_root(roots.pending.value[i],roots.pending.connected[i]);}
}
}
