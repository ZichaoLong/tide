#include "execution.h"
#include <tide/aggregate.h>
#include <tide/autograd.h>
#include <tide/delivery.h>
#include <tide/next.h>
#include <tide/read.h>
#include <tide/region.h>
#include <tide/state_evaluate.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <algorithm>
#include <stdexcept>

namespace accelerator_scale {
Resident::Resident(Graph graph, Model model, Options options, Index batch, Placement placement)
  : graph_(std::move(graph)), model_(std::move(model)), options_(options), placement_(placement),
    pool_(std::min<Index>(options.workers, placement.devices.size())) {
  graph_.compile(); configure_model(graph_, model_);
  validate_full_autograd(model_, options_); validate_aggregate_autograd(model_, options_);
  for (const auto& node : graph_.nodes)
    if (node.next_state != "adopt-v1" || (node.memory != "lh-add-repeat-v1"
        && node.memory != "lh-fiber-attention-all-softmax-repeat-v1"))
      throw std::invalid_argument("resident benchmark supports only its declared historical local programs");
  selection_model_ = model_;
  selection_model_.nodes[0].bias = at::zeros_like(model_.nodes[0].bias, at::TensorOptions().device(at::kCPU));
  for (const auto& region : graph_.regions)
    if (region.selector != "lh-count-affect-v1" && region.selector != "count-v1")
      throw std::invalid_argument("resident benchmark requires count-only CPU histories");
  for (size_t n = 0; n < model_.nodes.size(); ++n) {
    const auto& w = model_.nodes[n];
    w.kernel->validate_policy(graph_.nodes[n], graph_.source_counts[n]); w.kernel->validate_weights(w);
    w.full_kernel->validate_weights(w, graph_.outgoing_ports.offsets[n+1]-graph_.outgoing_ports.offsets[n]);
    w.aggregate_kernel->validate_weights(w, graph_.source_counts[n]);
    host_read_.push_back(w); host_read_.back().read = w.read.to(at::kCPU);
  }
  for (auto d : placement_.devices) {
    c10::impl::VirtualGuardImpl api(d.type()); streams_.push_back(api.getStream(d));
  }
  state_.identity = graph_.identity; state_.batch_size = batch;
}
void Resident::phase(const Groups& groups, const std::function<void(Index,const std::vector<size_t>&)>& fn) {
  std::vector<std::function<void()>> jobs;
  for (Index shard = 0; shard < static_cast<Index>(placement_.devices.size()); ++shard)
    jobs.push_back([&, shard] {
      c10::StreamGuard guard(streams_[shard]);
      for (const auto& [node, ids] : groups) if (placement_.node_device[node] == shard) fn(node, ids);
    });
  pool_.run(std::move(jobs));
  // One stream per shard; explicit phase barriers also cover cross-card sources.
  synchronize(placement_);
}
void Resident::read(std::vector<Event>& events, const std::vector<size_t>& ids) {
  const auto node = events[ids.front()].node;
  const auto& w = host_read_[node];
  const auto& mode = graph_.regions[graph_.nodes[node].region].read_mode;
  std::vector<Tensor> originals, numeric;
  for (auto i : ids) {
    const auto& e = events[i];
    auto request = read_input(mode, e.old, e.proposed_state, e.time, e.local_content());
    originals.push_back(request.state ? request.state->value : request.content.value);
  }
  auto evaluate = [&](const Tensor& value) {
    State state; state.value = value;
    return w.read_kernel->step(w, {&state, events[ids[0]].time, {value, {}, {}, {}, -1}});
  };
  {
    at::NoGradGuard guard;
    Transfer copy(at::Device(at::kCPU)); for (const auto& t : originals) copy.add(t); copy.execute();
    std::vector<State> states(originals.size()); std::vector<ReadInput> requests;
    for (size_t j = 0; j < ids.size(); ++j) {
      states[j].value = copy.get(originals[j]);
      requests.push_back({&states[j], events[ids[j]].time, {states[j].value, {}, {}, {}, -1}});
    }
    numeric = w.read_kernel->batch(w, requests);
  }
  for (size_t j = 0; j < ids.size(); ++j) {
    if (at::GradMode::is_enabled()) numeric[j] = semantic_value(numeric[j], evaluate(host(originals[j])));
    if (!numeric[j].device().is_cpu() || !at::isfinite(numeric[j]).item<bool>())
      throw std::runtime_error("invalid CPU descriptor");
    events[ids[j]].descriptor = numeric[j];
  }
}
AdvanceResult Resident::advance(const std::vector<External>& inputs, Index stop, Index seal) {
  if (failed_) throw std::runtime_error("resident executor failed; new instance required");
  if (state_.cut > stop || seal < stop) throw std::invalid_argument("invalid sealed window");
  auto ledger = state_.ledger;
  for (const auto& x : inputs) {
    if (x.batch < 0 || x.batch >= state_.batch_size || x.port < 0
        || x.port >= static_cast<Index>(graph_.inputs.size()) || x.position < 0
        || x.time < state_.cut || x.time >= stop || x.value.sizes() != at::IntArrayRef{model_.width()}
        || x.value.scalar_type() != at::kFloat || !at::isfinite(x.value).all().item<bool>())
      throw std::invalid_argument("invalid resident external record");
    Owner owner{x.batch,x.port}; auto it = ledger.find(owner);
    const auto previous = it == ledger.end() ? Owner{-1,-1} : it->second;
    if (x.position-1 != previous.first || x.time <= previous.second) throw std::invalid_argument("invalid resident input ledger");
    ledger[owner] = {x.position,x.time};
  }
  AdvanceResult result;
  try {
    state_.ledger = std::move(ledger);
    for (const auto& x : inputs) queue_[x.time].push_back({x.batch,graph_.inputs[x.port],x.time,0,x.port,x.position,x.value});
    while (!queue_.empty() && queue_.begin()->first < stop) {
      auto arrived = std::move(queue_.begin()->second); queue_.erase(queue_.begin());
      std::sort(arrived.begin(),arrived.end(),[](const auto& a,const auto& b){return a.key()<b.key();});
      std::vector<Event> events; Groups nodes; std::map<Owner,std::vector<size_t>> regions;
      for (auto& atom : arrived) {
        if (events.empty() || Owner{events.back().batch,events.back().node} != Owner{atom.batch,atom.node}) {
          Event e; e.batch=atom.batch; e.node=atom.node; e.time=atom.time;
          nodes[e.node].push_back(events.size()); regions[{e.batch,graph_.nodes[e.node].region}].push_back(events.size());
          events.push_back(std::move(e));
        }
        events.back().fiber.push_back(std::move(atom));
      }
      ++result.stats["logical_times"]; result.stats["candidate_events"] += events.size();
      result.stats["source_rows"] += arrived.size();
      phase(nodes,[&](Index node,const auto& ids) {
        const auto& w=model_.nodes[node]; Transfer transport(w.bias.device());
        for (auto i:ids) for (const auto& a:events[i].fiber) {
          transport.add(a.value);
          if (a.kind==1) {
            const auto bytes=a.value.numel()*a.value.element_size();
            if (a.value.device()==w.bias.device()) transfers.local_messages+=bytes;
            else transfers.remote_messages+=bytes;
          }
        }
        transport.execute();
        for (auto i:ids) {
          auto& e=events[i]; for (auto& a:e.fiber) a.value=transport.get(a.value);
          const auto old=state_.states.find({e.batch,node});
          e.old=old==state_.states.end()?w.kernel->initial(w):old->second;
        }
        auto content=evaluate_aggregate(graph_,model_,events,ids,true,options_.packed_sources,options_.aggregate_autograd);
        evaluate_state(model_,events,ids,true,content); read(events,ids);
      });
      struct RegionTask { Owner owner; const std::vector<size_t>* ids; const History* old; Selection selection; };
      std::vector<RegionTask> region_tasks;
      for(const auto& [owner,ids]:regions) {
        const auto old=state_.history.find(owner);
        region_tasks.push_back({owner,&ids,old==state_.history.end()?nullptr:&old->second,{}});
      }
      const auto region_workers=options_.parallel_regions?std::min<size_t>(options_.workers,region_tasks.size()):1;
      std::vector<std::function<void()>> select_jobs;
      for(size_t worker=0;worker<region_workers;++worker) select_jobs.push_back([&,worker]{
        for(size_t i=worker;i<region_tasks.size();i+=region_workers) {
          auto& task=region_tasks[i];task.selection=evaluate_selection(graph_,selection_model_,task.old,events,*task.ids);
        }
      });
      pool_.run(std::move(select_jobs));
      for(auto& task:region_tasks) {
        const auto& owner=task.owner;const auto& ids=*task.ids;
        commit_selection(state_,owner,std::move(task.selection),events,ids,options_.trace);
        for (auto i:ids) {
          auto& e=events[i]; e.comparison_state=graph_.regions[owner.second].observe_all||e.active?e.proposed_state:e.old;
          e.comparison=e.comparison_state.value;
        }
      }
      phase(nodes,[&](Index node,const auto& all) {
        const auto& w=model_.nodes[node]; Transfer controls(w.bias.device());
        for(auto i:all) controls.add(events[i].control); controls.execute();
        std::vector<size_t> active; std::vector<NextInput> requests;
        for(auto i:all) {
          auto& e=events[i]; e.control=controls.get(e.control);
          requests.push_back({e.old,e.comparison_state,e.time,e.local_content(),e.active,e.control});
          if(e.active) active.push_back(i);
        }
        std::vector<State> next;
        if(options_.batch_next)next=evaluate_next_batch(graph_.nodes[node],w,requests);
        else for(const auto& r:requests)next.push_back(evaluate_next(graph_.nodes[node],w,r));
        for(size_t j=0;j<all.size();++j) {events[all[j]].next_state=std::move(next[j]);events[all[j]].next=events[all[j]].next_state.value;}
        evaluate_full(graph_,model_,events,active,options_,true);
      });
      for(auto& e:events) {
        state_.states[{e.batch,e.node}]=e.next_state;
        if(e.active) {
          ++result.stats["selected_events"];
          c10::StreamGuard guard(streams_[placement_.node_device[e.node]]);
          deliver(graph_,model_,e,[&](const Atom& a){queue_[a.time].push_back(a);if(options_.trace)result.messages.push_back(a);++result.stats["visited_edges"];},
            [&](const Output& output){result.outputs.push_back(output);});
        }
        if(options_.trace)result.trace.push_back(std::move(e));
      }
      synchronize(placement_);
    }
    state_.cut=stop; result.cut=stop;
    result.stats["cached_states"]=state_.states.size(); result.stats["external_records"]=inputs.size();
    std::sort(result.messages.begin(),result.messages.end(),[&](const auto& a,const auto& b){
      return std::tie(a.position,a.batch,graph_.edges[a.source].source,a.source)<std::tie(b.position,b.batch,graph_.edges[b.source].source,b.source);});
    std::sort(result.outputs.begin(),result.outputs.end(),[](const auto& a,const auto& b){return std::tie(a.time,a.batch,a.port)<std::tie(b.time,b.batch,b.port);});
    return result;
  } catch(...) {failed_=true;throw;}
}
Continuation Resident::snapshot() const {
  if(failed_)throw std::runtime_error("resident executor failed");
  auto q=state_; export_pending(q,queue_);
  for(auto& [owner,s]:q.states){s.value=s.value.clone();for(auto& [name,t]:s.slots)t=t.clone();}
  for(auto& a:q.pending)a.value=a.value.clone();
  return q;
}
Execution::Execution(Graph graph,Model model,Options options,Index batch,const Placement& placement) {
  if(placement.resident)resident_=std::make_unique<Resident>(std::move(graph),std::move(model),options,batch,placement);
  else {
    host_=std::make_unique<Streaming>(std::move(graph),std::move(model),options);
    Continuation q;q.identity=host_->graph().identity;q.batch_size=batch;
    cursor_=std::make_unique<StreamingCursor>(*host_,q);
  }
}
AdvanceResult Execution::advance(const std::vector<External>& x,Index stop,Index seal){return resident_?resident_->advance(x,stop,seal):cursor_->advance(x,stop,seal);}
Continuation Execution::snapshot() const{return resident_?resident_->snapshot():cursor_->snapshot();}
const std::string& Execution::identity() const{return resident_?resident_->identity():host_->graph().identity;}
}  // namespace accelerator_scale
