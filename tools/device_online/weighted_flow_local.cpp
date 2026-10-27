#include "weighted_flow_internal.h"
namespace tide::device_online {
void WeightedFlow::Impl::append_nodes(DeviceProgram& p,const ReadyBatch& ready){
  const auto samples=boundary.batch_size;
  for(size_t v=0;v<nodes.size();++v){auto& n=*nodes[v];
    n.input=append_weighted_node_input(p,ready,n.source.classes,v,samples,graph.inputs.size(),error);
    const auto eps=std::visit([](const auto& c){return c.epsilon;},n.binding.memory);
    auto& aggregate=services->begin();
    n.aggregate=append_weighted_aggregate(aggregate,n.input.payload,n.input.lengths,n.input.owners,n.source.values,
      n.binding.aggregate_alpha,eps,error,remaining(aggregate));services->end(aggregate);
    n.mass=weighted_columns(p,n.aggregate->output,0,1).reshape({samples});n.content=weighted_columns(p,n.aggregate->output,1,model.width()-1);
    weighted_copy(p,n.old.floating,n.state.floating);weighted_copy(p,n.old.integer,n.state.integer);
    auto& update=services->begin();
    if(auto c=std::get_if<weighted::KVConfig>(&n.binding.memory)){
      n.kv_update=append_weighted_kv_update(update,*c,n.old,n.aggregate->output,n.input.times,n.input.bank,n.input.present,
        bind_weighted_kv_update(n.banks),error,remaining(update));n.proposal=n.kv_update->raw.result;
    }else{
      const auto& rc=std::get<weighted::RecurrentConfig>(n.binding.memory);n.recurrent_parameters=bind_weighted_recurrent(n.banks,rc);
      n.recurrent_update=append_weighted_recurrent(update,rc,WeightedRecurrentAction::Update,n.old,n.input.times,n.mass,n.content,
        n.input.bank,n.input.present,n.recurrent_parameters,error,remaining(update));n.proposal=n.recurrent_update->update->result;
    }
    services->end(update);
    const auto& mode=graph.regions[graph.nodes[v].region].read_mode;
    const auto read=mode=="content"?n.initial:mode=="old"?n.old:n.proposal;
    WeightedDescribeInput x;x.mass=n.mass;x.content=n.content;int64_t summary=0;
    auto& mem=services->begin();
    if(auto c=std::get_if<weighted::KVConfig>(&n.binding.memory)){
      summary=c->summary_width;at::Tensor memory,total;
      if(c->memory_read=="sum"){
        n.kv_sum=append_weighted_kv_sum(mem,*c,read,n.input.times,n.input.present,error,remaining(mem));memory=n.kv_sum->memory;total=n.kv_sum->total;
      }else{
        n.kv_query=append_weighted_kv_query(mem,*c,read,n.input.times,n.content,n.input.bank,n.input.present,
          bind_weighted_kv_query(n.banks),error,remaining(mem));memory=n.kv_query->memory;total=n.kv_query->total;
      }
      x.short_memory=weighted_columns(mem,memory.reshape({samples,2*summary}),0,summary);
      x.long_memory=weighted_columns(mem,memory.reshape({samples,2*summary}),summary,summary);
      x.short_total=weighted_columns(mem,total,0,1).reshape({samples});x.long_total=weighted_columns(mem,total,1,1).reshape({samples});
    }else{
      const auto& rc=std::get<weighted::RecurrentConfig>(n.binding.memory);summary=rc.summary_width;
      n.recurrent_mem=append_weighted_recurrent(mem,rc,WeightedRecurrentAction::Memory,read,n.input.times,n.mass,n.content,
        n.input.bank,n.input.present,n.recurrent_parameters,error,remaining(mem));const auto& f=n.recurrent_mem->fields;
      x.short_memory=f.at("short_memory");x.long_memory=f.at("long_memory");
      x.short_total=f.at("short_total").reshape({samples});x.long_total=f.at("long_total").reshape({samples});
    }
    services->end(mem);
    auto& describe=services->begin();
    n.describe=append_weighted_describe(describe,x,n.input.bank,bind_weighted_describe(n.banks,model.width()-1,summary,n.binding.regional),
      n.binding.regional,eps,error,remaining(describe));services->end(describe);
  }
}
void WeightedFlow::Impl::append_regions(DeviceProgram& p){
  const auto samples=boundary.batch_size;const auto opts=at::TensorOptions().dtype(at::kFloat).device(device);
  for(auto& ptr:regions){auto& r=*ptr;const auto members=static_cast<int64_t>(r.nodes.size());
    const auto e=r.config.regional()?*r.config.region_control_width:1,q=r.config.region_score=="att"?*r.config.selection_width:1,width=3+e+2*q;
    auto desc=at::zeros({samples,members,width},opts),time=at::zeros({samples,members},opts.dtype(at::kLong)),present=at::zeros({samples,members},opts.dtype(at::kBool));
    for(int64_t j=0;j<members;++j){auto& n=*nodes[r.nodes[j]];const auto& f=n.describe->fields;
      auto target=desc.select(1,j);weighted_copy(p,target.narrow(1,0,1),f.at("w"));weighted_copy(p,target.narrow(1,1,1),f.at("mu"));weighted_copy(p,target.narrow(1,2,1),f.at("score"));
      if(f.count("evidence"))weighted_copy(p,target.narrow(1,3,e),f.at("evidence"));
      if(f.count("x")){weighted_copy(p,target.narrow(1,3+e,q),f.at("x"));weighted_copy(p,target.narrow(1,3+e+q,q),f.at("key"));}
      weighted_copy(p,time.select(1,j),n.input.times);weighted_copy(p,present.select(1,j),n.input.present);
    }
    r.input=append_weighted_region_input(p,desc,present,time,r.nodes,r.config,error);
    auto& selection=services->begin();
    r.selection=append_weighted_region(selection,r.input.fields,r.input.ids,r.input.times,r.input.lengths,bind_weighted_region(r.banks,r.config,samples),r.config,error,remaining(selection));
    services->end(selection);
    auto control=at::zeros({samples*members+1},opts),active=at::zeros({samples*members+1},opts.dtype(at::kBool));
    weighted_copy(p,control.narrow(0,0,samples*members),r.selection->controls.reshape({-1}));weighted_copy(p,active.narrow(0,0,samples*members),r.selection->active.reshape({-1}));
    auto controls=at::zeros({samples,members},opts),actives=at::zeros({samples,members},opts.dtype(at::kBool));
    p.index_select(control,0,r.input.inverse.reshape({-1}),controls.reshape({-1}));p.index_select(active,0,r.input.inverse.reshape({-1}),actives.reshape({-1}));
    for(int64_t j=0;j<members;++j){auto& n=*nodes[r.nodes[j]];n.control=weighted_columns(p,controls,j,1).reshape({samples});n.active=weighted_columns(p,actives,j,1).reshape({samples});}
  }
}
ActionBatch WeightedFlow::Impl::append_full(DeviceProgram& p){
  const auto samples=boundary.batch_size,count=static_cast<int64_t>(nodes.size()),width=model.width();
  const auto opts=at::TensorOptions().dtype(at::kFloat).device(device);
  ActionBatch out{at::zeros({count*samples,4},opts.dtype(at::kLong)),at::zeros({count*samples,width},opts),at::zeros({count*samples},opts.dtype(at::kBool))};
  for(int64_t v=0;v<count;++v){auto& n=*nodes[v];
    auto& full=services->begin();
    if(auto c=std::get_if<weighted::KVConfig>(&n.binding.memory)){
      n.kv_full=append_weighted_kv_full(full,*c,n.proposal,n.input.times,n.content,n.input.bank,n.active,bind_weighted_kv_full(n.banks),error,remaining(full));n.full=n.kv_full->output;
    }else{
      n.recurrent_full=append_weighted_recurrent(full,std::get<weighted::RecurrentConfig>(n.binding.memory),WeightedRecurrentAction::Full,
        n.proposal,n.input.times,n.mass,n.content,n.input.bank,n.active,n.recurrent_parameters,error,remaining(full));n.full=n.recurrent_full->fields.at("G");
    }
    services->end(full);
    auto meta=out.coordinates.narrow(0,v*samples,samples);weighted_copy(p,meta.select(1,0),at::arange(samples,meta.options()));weighted_copy(p,meta.select(1,1),at::full({samples},v,meta.options()));
    weighted_copy(p,meta.select(1,2),n.input.times);weighted_copy(p,meta.select(1,3),n.input.times);
    weighted_copy(p,out.values.narrow(0,v*samples,samples).narrow(1,0,1),n.control.reshape({samples,1}));
    weighted_copy(p,out.values.narrow(0,v*samples,samples).narrow(1,1,width-1),n.full);weighted_copy(p,out.valid.narrow(0,v*samples,samples),n.active);
  }
  return out;
}
}
