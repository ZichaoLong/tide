#include "state_owner.h"
#include <algorithm>
#include <stdexcept>

namespace tide::device_online {
namespace {
StateKernelProfile cpu_view(const ContentProfile& p,const std::vector<int64_t>& owned) {
  StateKernelProfile out(p);out.nodes.clear();out.weights.clear();out.source_counts.clear();
  int64_t previous=-1;
  for(const auto n:owned) {
    if(n<=previous||n>=int64_t(p.graph.nodes.size()))throw std::invalid_argument("invalid state owner nodes");
    previous=n;out.nodes.push_back(p.graph.nodes[n]);out.weights.push_back(p.model.nodes[n]);out.source_counts.push_back(p.graph.source_counts[n]);
  }
  return out;
}
long double common_bytes(const StateKernelProfile& p,const Continuation& q,const ContentLimits& l) {
  // State/proposal/Read scratch, compact parameter tables and per-stage journals.
  return 64.L*q.batch_size*p.nodes.size()*(p.width+4.L)+64.L*p.nodes.size()*(p.width+8.L)
    +64.L*l.queue*(p.width+16.L)+p.sources.nbytes()+4096;
}
}
long double StateOwner::minimum_bytes(const ContentProfile& p,const std::vector<int64_t>& owned,
    const Continuation& q,const ContentLimits& l) {
  const auto view=cpu_view(p,owned);if(owned.empty())return 0;
  return common_bytes(view,q,l)+PackedFiberAttention::minimum_bytes(view,q,l)+PackedEventAttention::minimum_bytes(view,q,l);
}
StateOwner::StateOwner(const ContentProfile& p,std::vector<int64_t> owned,const Continuation& q,at::Device d,
    ContentLimits l,int64_t budget):global_nodes_(std::move(owned)),profile_([&] {
      if(global_nodes_.empty()||minimum_bytes(p,global_nodes_,q,l)>budget)
        throw std::invalid_argument("state owner is empty or exceeds tensor budget");
      return StateKernelProfile(p,global_nodes_,d);
    }()),limits_(l) {
  auto initial=state_shard_initial(q,global_nodes_);const auto nodes=int64_t(global_nodes_.size());
  auto opts=at::TensorOptions().dtype(profile_.dtype);
  auto values=at::zeros({q.batch_size,nodes,profile_.width},opts),clocks=at::zeros({q.batch_size,nodes,2},at::kLong);
  auto present=at::zeros({q.batch_size,nodes},at::kBool);clocks.select(2,0).fill_(-1);
  for(const auto& [owner,s]:initial.states) {
    const auto [b,n]=owner;values[b][n].copy_(s.value);clocks[b][n][0].fill_(s.last_time);
    clocks[b][n][1].fill_(s.observations);present[b][n].fill_(true);
  }
  state_={values.to(d),clocks.to(d),present.to(d)};coefficients_=at::empty_like(profile_.decay);
  event_count_=at::zeros({1},clocks.options().device(d));
  const auto common=common_bytes(profile_,q,l),fiber=PackedFiberAttention::minimum_bytes(profile_,initial,l),
    event=PackedEventAttention::minimum_bytes(profile_,initial,l);
  const auto extra=(budget-common-fiber-event)/std::max<int64_t>(1,int64_t(fiber>0)+int64_t(event>0));
  reserved_=int64_t(common);
  if(fiber>0){fiber_=std::make_unique<PackedFiberAttention>(profile_,initial,d,l,int64_t(fiber+extra));reserved_+=fiber_->reserved_bytes();}
  if(event>0){event_=std::make_unique<PackedEventAttention>(profile_,initial,d,l,int64_t(event+extra));reserved_+=event_->reserved_bytes();}
  if(reserved_>budget)throw std::logic_error("state owner allocation exceeded admission");
}
StateReadProposal StateOwner::append_read(CannProgram& p,const ReadyBatch& ready,const ContentBatch& content,const at::Tensor& error) {
  StateReadProposal out;p.sigmoid(profile_.decay,coefficients_);
  if(fiber_){out.fiber=fiber_->propose(p,profile_,ready,content,state_,error);out.values=out.fiber.values;}
  if(event_){out.event=event_->propose(p,ready,content,error,out.values);out.values=out.event.values;}
  device_online::append_read(p,profile_,ready,content,state_,coefficients_,error,limits_.max_repeat_ticks,limits_.vectorized_read,out.values);
  return out;
}
ContentUpdate StateOwner::append_update(CannProgram& p,const ReadyBatch& ready,const ContentBatch& content,
    const SelectionProposal& selection,const StateReadProposal& read,const at::Tensor& stage,const at::Tensor& error) {
  return append_content_state(p,profile_,ready,content,selection,state_,coefficients_,stage,event_count_,error,limits_,read.values);
}
void StateOwner::append_commit(CannProgram& p,const ContentUpdate& update,const StateReadProposal& read,
    const SelectionProposal& selection,const at::Tensor& error) {
  commit_content_state(p,state_,update,error);
  if(fiber_)fiber_->commit(p,read.fiber,selection,error);
  if(event_)event_->commit(p,read.event,selection,error);
}
void StateOwner::reset_window(){event_count_.zero_();if(fiber_)fiber_->reset_window();if(event_)event_->reset_window();}
void StateOwner::export_states(Continuation& q) const {
  Continuation local;local.batch_size=q.batch_size;local.cut=q.cut;
  const auto values=state_.values.cpu(),clocks=state_.clocks.cpu(),present=state_.present.cpu();
  for(int64_t b=0;b<q.batch_size;++b)for(size_t n=0;n<global_nodes_.size();++n)if(present[b][n].item<bool>())
    local.states[{b,int64_t(n)}]={values[b][n].clone(),clocks[b][n][0].item<int64_t>(),clocks[b][n][1].item<int64_t>()};
  if(fiber_)fiber_->export_states(local);if(event_)event_->export_states(local);
  for(auto& [owner,state]:local.states)q.states[{owner.first,global_nodes_[owner.second]}]=std::move(state);
}
void StateOwner::export_trace(std::vector<Event>& events) const {
  std::vector<Event> local;std::vector<size_t> rows;
  for(size_t i=0;i<events.size();++i) {
    const auto it=std::lower_bound(global_nodes_.begin(),global_nodes_.end(),events[i].node);
    if(it==global_nodes_.end()||*it!=events[i].node)continue;
    local.push_back(events[i]);local.back().node=it-global_nodes_.begin();rows.push_back(i);
  }
  if(fiber_)fiber_->export_trace(local);if(event_)event_->export_trace(local);
  for(size_t i=0;i<rows.size();++i){local[i].node=global_nodes_[local[i].node];events[rows[i]]=std::move(local[i]);}
}
std::map<std::string,int64_t> StateOwner::stats() const {
  std::map<std::string,int64_t> out{{"state_shard_state_bytes",int64_t(state_.values.nbytes()+state_.clocks.nbytes()+state_.present.nbytes())}};
  for(const bool event:{false,true}) {
    const std::string prefix=event?"event_attention_":"attention_";const bool has=event?bool(event_):bool(fiber_);
    out[prefix+"chunks"]=has?(event?event_->chunks().item<int64_t>():fiber_->chunks().cpu().item<int64_t>()):0;
    out[prefix+"chunk_rows"]=has?(event?event_->chunk_rows():fiber_->chunk_rows()):0;
    out[prefix+"kv_peak"]=has?(event?event_->peak().item<int64_t>():fiber_->peak().cpu().item<int64_t>()):0;
    out[prefix+"kv_capacity"]=has?limits_.kv_rows:0;
    out[prefix+"key_rows"]=has?(event?event_->key_rows():fiber_->key_rows()):0;
    const auto work=has?(event?event_->key_work():fiber_->key_work().cpu()):at::zeros({3},at::kLong);
    out[prefix+"key_tiles"]=work[0].item<int64_t>();out[prefix+"tiled_score_entries"]=work[1].item<int64_t>();
    out[prefix+"tiled_padding_entries"]=work[2].item<int64_t>();
  }
  return out;
}
} // namespace tide::device_online
