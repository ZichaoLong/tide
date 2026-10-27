#include "weighted_flow_internal.h"
#include <algorithm>
namespace tide::device_online {
void WeightedFlow::Impl::save_stage(DeviceProgram& p,const ReadyBatch& ready,const AtomBatch& arrivals){
  auto save=[&](const std::string& name,const at::Tensor& x){journal->save(p,stages,name,x);};
  save("fiber.meta",ready.atoms.coordinates);save("fiber.values",ready.atoms.values);save("fiber.present",ready.atoms.valid);
  save("fiber.rows",ready.fibers);save("fiber.offsets",ready.fiber_offsets);
  save("message.meta",arrivals.coordinates);save("message.values",arrivals.values);save("message.present",arrivals.valid);
  save("counts",ready.counts);
  for(size_t v=0;v<nodes.size();++v){const auto& n=*nodes[v];const auto prefix="node."+std::to_string(v)+".";
    save(prefix+"present",n.input.present);save(prefix+"times",n.input.times);save(prefix+"content",n.aggregate->output);
    save(prefix+"old.float",n.old.floating);save(prefix+"old.int",n.old.integer);
    save(prefix+"proposal.float",n.proposal.floating);save(prefix+"proposal.int",n.proposal.integer);
    save(prefix+"control",n.control);save(prefix+"active",n.active);save(prefix+"full",n.full);
    for(const auto& [name,x]:n.describe->fields)save(prefix+"describe."+name,x);
  }
  for(size_t r=0;r<regions.size();++r){const auto& t=*regions[r]->selection;
    save("region."+std::to_string(r)+".times",t.history_times);save("region."+std::to_string(r)+".present",t.history_present);
  }
}
namespace {
std::vector<Atom> atoms(const at::Tensor& coords,const at::Tensor& values,const at::Tensor& valid){
  std::vector<Atom> out;
  for(int64_t i=0;i<valid.numel();++i)if(valid[i].item<bool>()){auto c=coords[i];out.push_back({c[0].item<int64_t>(),c[1].item<int64_t>(),c[2].item<int64_t>(),
    c[3].item<int64_t>(),c[4].item<int64_t>(),c[5].item<int64_t>(),values[i].clone()});}return out;
}
}
Result WeightedFlow::Impl::decode()const{
  Result result;result.continuation=checkpoint();const auto count=stages.item<int64_t>();auto record=journal->decode(count);
  auto get=[&](const std::string& name,int64_t t){return record.at(name)[t];};
  int64_t fibers=0,received=0,fulls=0;
  for(int64_t t=0;t<count;++t){
    auto source=atoms(get("fiber.meta",t),get("fiber.values",t),get("fiber.present",t));
    auto sent=atoms(get("message.meta",t),get("message.values",t),get("message.present",t));result.messages.insert(result.messages.end(),sent.begin(),sent.end());
    received+=get("counts",t)[0].item<int64_t>();fibers+=get("counts",t)[1].item<int64_t>();
    for(size_t v=0;v<nodes.size();++v){const auto& n=*nodes[v];const auto prefix="node."+std::to_string(v)+".";
      auto old=decode_weighted_raw_state(n.layout,{get(prefix+"old.float",t),get(prefix+"old.int",t)});
      auto proposal=decode_weighted_raw_state(n.layout,{get(prefix+"proposal.float",t),get(prefix+"proposal.int",t)});
      for(int64_t b=0;b<boundary.batch_size;++b)if(get(prefix+"present",t)[b].item<bool>()){
        Event e;e.batch=b;e.node=v;e.time=get(prefix+"times",t)[b].item<int64_t>();
        for(const auto& atom:source)if(atom.batch==b&&atom.node==int64_t(v)&&atom.time==e.time)e.fiber.push_back(atom);
        for(const auto& atom:e.fiber){auto visible=atom;
          if(atom.kind==1&&!graph.origins.empty()&&graph.origin_index[atom.source]>=0){const auto& origin=graph.origins[graph.origin_index[atom.source]];
            visible.kind=0;visible.source=origin.port;visible.position/=origin.stride;}
          const auto slot=atom.kind==0?graph.source_domain->input[atom.source]:graph.source_domain->edge_target[atom.source];
          e.sources.push_back({slot,visible,at::ones({},atom.value.options())});
        }
        if(!graph.origins.empty())std::stable_sort(e.sources.begin(),e.sources.end(),[](const SourceInput& a,const SourceInput& b){return a.atom.key()<b.atom.key();});
        e.content=get(prefix+"content",t)[b].clone();e.old=std::move(old[b]);e.proposed_state=std::move(proposal[b]);
        e.comparison_state=e.next_state=e.proposed_state;e.proposal=e.proposed_state.value;e.comparison=e.comparison_state.value;e.next=e.next_state.value;
        e.control=get(prefix+"control",t)[b].clone();e.active=get(prefix+"active",t)[b].item<bool>();
        for(const auto& [name,_]:n.describe->fields){auto x=get(prefix+"describe."+name,t)[b].clone();
          e.descriptor_fields[name]=(name=="w"||name=="mu"||name=="score")?x.reshape({}):x;}
        const auto region=graph.nodes[v].region;e.history.last_time=get("region."+std::to_string(region)+".times",t)[b].item<int64_t>();
        if(e.active){++fulls;e.full=at::cat({e.control.reshape({1}),get(prefix+"full",t)[b]});
          const auto start=graph.outgoing_ports.offsets[v],end=graph.outgoing_ports.offsets[v+1];
          for(auto slot=start;slot<end;++slot){e.emitted.push_back({slot-start,e.full});const auto binding=graph.outgoing_ports.bindings[slot];
            if(binding.kind==0)result.outputs.push_back({b,e.time,binding.id,e.full});}
        }
        result.trace.push_back(std::move(e));
      }
    }
  }
  std::sort(result.trace.begin(),result.trace.end(),[](const Event& a,const Event& b){return std::tie(a.time,a.batch,a.node)<std::tie(b.time,b.batch,b.node);});
  std::sort(result.messages.begin(),result.messages.end(),[&](const Atom& a,const Atom& b){return std::tie(a.position,a.batch,graph.edges[a.source].source,a.source)<std::tie(b.position,b.batch,graph.edges[b.source].source,b.source);});
  std::sort(result.outputs.begin(),result.outputs.end(),[&](const Output& a,const Output& b){return std::tie(a.time,a.batch,graph.outputs[a.port],a.port)<std::tie(b.time,b.batch,graph.outputs[b.port],b.port);});
  result.stats={{"resident_stages",count},{"candidate_updates",fibers},{"full_evaluations",fulls},{"received_atoms",received},
    {"messages",static_cast<int64_t>(result.messages.size())},{"one_time_per_region",1},{"resident_tensor_bytes",bytes},
    {"physical_local_rows",count*boundary.batch_size*static_cast<int64_t>(nodes.size())},
    {"physical_full_rows",count*boundary.batch_size*static_cast<int64_t>(nodes.size())},
    {"operator_workspace_bytes",services->workspace_bytes()}};return result;
}
}
