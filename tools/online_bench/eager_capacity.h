#pragma once
#include "eager_placement.h"
#include "traffic_bounds.h"

namespace tide_flow::eager_capacity {
constexpr Index MIB=1024*1024;
struct Card {
  Index device=0,peak=0,budget=0,usable=0;
  std::map<std::string,Index> components,phases;
};
struct Attempt {Index rows,head;std::vector<Index> peaks;bool accepted;};
struct Plan {
  bool accepted=false;Index requested=0,rows=0,positions=0,connected=0,head_usable=0;
  EagerPlacement placement;Traffic traffic;std::vector<Card> cards;std::vector<Attempt> attempts;
};
inline std::vector<Card> envelope(const Packet& p,const Config& c,const Plan& plan,Index rows) {
  const auto& q=plan.traffic;const auto& owners=plan.placement.owners;
  const Index width=p.width,vocab=p.vocab,batch=p.batch,n=p.body.nodes.size();
  const Index payload=c.runtime.dtype==at::kDouble?8:c.runtime.dtype==at::kHalf?2:4;
  const bool attention=p.memory=="attention";std::vector<Card> cards;
  for(Index device=0;device<c.devices;++device) {
    Index nodes=0,max_events=0,max_keys=0;
    for(Index i=0;i<n;++i)if(owners[i]==device){++nodes;max_events=std::max(max_events,q.node_events[i]);max_keys=std::max(max_keys,q.node_atoms[i]);}
    max_events=product({max_events,plan.connected});max_keys=product({max_keys,plan.positions});
    const auto atoms=q.owner_atoms[device],events=q.owner_events[device],emissions=q.owner_emissions[device];
    const auto learned=product({payload,plan.placement.elements[device]});
    const auto constants=sum({product({payload,sum({product({2,width,width}),product({16,width}),16})}),product({4096,nodes})});
    Index persistent=product({batch,nodes,sum({product({payload,width}),4096})});
    const auto cache=attention?product({atoms,plan.positions,sum({product({2,width}),1}),payload}):0;
    persistent=sum({persistent,product({batch,cache})});
    const auto optimizer=c.training?product({c.optimizer=="adamw"?3:2,learned}):0;
    const auto vector_work=product({rows,plan.connected,sum({
        product({payload,width,sum({product({64,atoms}),product({32,events}),product({32,emissions})})}),
        product({8192,sum({atoms,events,emissions})})})});
    const auto cache_work=attention?product({rows,cache,c.training?product({4,sum({max_events,2})}):4}):0;
    const auto score_work=attention?product({rows,atoms,plan.connected,max_keys,4,16}):0;
    const auto largest_body=product({attention?3:1,width,width});
    const auto largest=std::max(largest_body,device==0?product({width,vocab}):0);
    const auto operator_work=sum({product({8,payload,largest_body,c.workers}),product({4,payload,largest}),64*MIB});
    Index head_work=0;
    if(device==0) {
      const auto outputs=product({rows,plan.connected,q.output_frames});
      head_work=c.training?sum({product({8,payload,outputs,vocab}),product({8,payload,outputs,width}),product({3,payload,vocab,width})})
                          :sum({product({4,payload,outputs,vocab}),product({2,payload,outputs,width})});
      head_work=sum({head_work,64*MIB});
      persistent=sum({persistent,product({batch,payload,width,2}),product({batch,8192})});
    }
    const Index transport=16*MIB,backend=512*MIB;
    const auto base=sum({learned,constants,persistent,backend});
    const auto forward=sum({base,optimizer,vector_work,cache_work,score_work,operator_work,head_work,transport});
    Card card;card.device=device;
    card.phases={{"construction",sum({learned,constants,backend,product({std::max<Index>(32,8*payload),largest})})},
                 {"forward",forward},{"backward",c.training?forward:0},{"optimizer",c.training?sum({base,optimizer,operator_work}):0}};
    card.components={{"learned",learned},{"constants",constants},{"persistent_state_and_kv",persistent},
        {"gradients_and_optimizer_slots",optimizer},{"vector_work",vector_work},{"cache_work",cache_work},
        {"attention_scores",score_work},{"operator_work",operator_work},{"head_work",head_work},
        {"transport",transport},{"backend_allowance",backend}};
    for(const auto& [_,v]:card.phases)card.peak=std::max(card.peak,v);
    cards.push_back(std::move(card));
  }return cards;
}
inline Plan plan(const Packet& p,const Config& c,const std::vector<Index>& budgets) {
  if((c.runtime.dtype!=at::kFloat&&c.runtime.dtype!=at::kDouble&&c.runtime.dtype!=at::kHalf)||(c.runtime.dtype==at::kHalf&&c.training))
    throw std::invalid_argument("eager capacity requires a supported inference dtype or FP32/FP64 training");
  if(Index(budgets.size())!=c.devices||budgets.empty()||*std::min_element(budgets.begin(),budgets.end())<1
      ||c.head_workspace_bytes<1||c.steps<1||c.warmup<0||c.windows<1||c.workers<1||c.sample_chunk_rows<0
      ||(c.chunk_policy!="conservative"&&c.chunk_policy!="aggressive")||(c.optimizer!="sgd"&&c.optimizer!="adamw"))
    throw std::invalid_argument("invalid eager capacity geometry/budget");
  Plan result;result.placement=eager_placement(p,c);result.traffic=traffic_bounds(p,result.placement.owners);
  result.positions=product({sum({c.steps,c.warmup}),c.windows,p.tokens});result.connected=product({c.windows,p.tokens});
  result.requested=result.rows=std::min(c.sample_chunk_rows?c.sample_chunk_rows:p.batch,p.batch);
  const auto divisor=c.chunk_policy=="aggressive"?10:4;
  std::vector<Index> usable;for(auto b:budgets)usable.push_back(std::max<Index>(0,b-b/divisor-128*MIB));
  result.head_usable=c.head_workspace_bytes-c.head_workspace_bytes/divisor;
  while(true) {
    result.cards=envelope(p,c,result,result.rows);Attempt attempt{result.rows,result.cards[0].components.at("head_work"),{},true};
    for(Index i=0;i<c.devices;++i){attempt.peaks.push_back(result.cards[i].peak);attempt.accepted&=result.cards[i].peak<=usable[i];}
    attempt.accepted&=attempt.head<=result.head_usable;
    result.attempts.push_back(attempt);result.accepted=attempt.accepted;
    if(result.accepted||!c.auto_sample_chunks||result.rows==1)break;
    result.rows=std::max<Index>(1,result.rows/2);
  }
  for(Index i=0;i<c.devices;++i){result.cards[i].budget=budgets[i];result.cards[i].usable=usable[i];}
  return result;
}
inline void record(std::ostream& out,const Packet& p,const Config& c,const Plan& plan) {
  out<<"{\"schema\":\"tide-eager-capacity-v1\",\"state\":"<<quoted(plan.accepted?"admitted":"refused")
      <<",\"scope\":\"static finite-run envelope for declared eager consumers; allocator calibration required\",\"requested_sample_rows\":"<<plan.requested
      <<",\"effective_sample_rows\":"<<plan.rows<<",\"logical_batch\":"<<p.batch<<",\"physical_chunks\":"<<(p.batch-1)/plan.rows+1
      <<",\"sample_reductions\":"<<plan.attempts.size()-1<<",\"head_workspace_bytes\":"<<c.head_workspace_bytes
      <<",\"head_usable_bytes\":"<<plan.head_usable<<",\"chunk_policy\":"<<quoted(c.chunk_policy)<<",\"node_owners\":";
  array(out,plan.placement.owners);out<<",\"parameter_elements\":";array(out,plan.placement.elements);
  out<<",\"positions\":"<<plan.positions<<",\"connected_positions\":"<<plan.connected<<",\"traffic\":";record(out,plan.traffic);
  out<<",\"attempts\":[";
  for(size_t i=0;i<plan.attempts.size();++i){if(i)out<<',';const auto& a=plan.attempts[i];
    out<<"{\"physical_rows\":"<<a.rows<<",\"estimated_peak_bytes\":";array(out,a.peaks);
    out<<",\"head_bytes\":"<<a.head<<",\"accepted\":"<<(a.accepted?"true":"false")<<'}';}
  out<<"],\"devices\":[";
  for(size_t i=0;i<plan.cards.size();++i){if(i)out<<',';const auto& d=plan.cards[i];
    out<<"{\"logical_device\":"<<d.device<<",\"estimated_peak_bytes\":"<<d.peak<<",\"budget_bytes\":"<<d.budget<<",\"usable_bytes\":"<<d.usable<<",\"components\":";
    fields(out,d.components);out<<",\"phases\":";fields(out,d.phases);out<<'}';}
  out<<"]}";
}
} // namespace tide_flow::eager_capacity
