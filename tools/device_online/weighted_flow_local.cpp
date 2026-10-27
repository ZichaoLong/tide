#include "weighted_flow_internal.h"
namespace tide::device_online {
void append_weighted_node_local(DeviceProgram& p,TaskServices& services,WeightedFlowNode& n,const std::string& mode,
  const at::Tensor& error,const WeightedFlowBudget& remaining){
  const auto samples=n.input.present.numel();
  const auto eps=std::visit([](const auto& c){return c.epsilon;},n.binding.memory);
  auto& aggregate=services.begin();
  n.aggregate=append_weighted_aggregate(aggregate,n.input.payload,n.input.lengths,n.input.owners,n.source.values,
    n.binding.aggregate_alpha,eps,error,remaining(aggregate));services.end(aggregate);
  n.mass=weighted_columns(p,n.aggregate->output,0,1).reshape({samples});n.content=weighted_columns(p,n.aggregate->output,1,n.input.payload.size(2)-1);
  weighted_copy(p,n.old.floating,n.state.floating);weighted_copy(p,n.old.integer,n.state.integer);
  auto& update=services.begin();
  if(auto c=std::get_if<weighted::KVConfig>(&n.binding.memory)){
    n.kv_update=append_weighted_kv_update(update,*c,n.old,n.aggregate->output,n.input.times,n.input.bank,n.input.present,
      bind_weighted_kv_update(n.banks),error,remaining(update));n.proposal=n.kv_update->raw.result;
  }else{
    const auto& rc=std::get<weighted::RecurrentConfig>(n.binding.memory);n.recurrent_parameters=bind_weighted_recurrent(n.banks,rc);
    n.recurrent_update=append_weighted_recurrent(update,rc,WeightedRecurrentAction::Update,n.old,n.input.times,n.mass,n.content,
      n.input.bank,n.input.present,n.recurrent_parameters,error,remaining(update));n.proposal=n.recurrent_update->update->result;
  }
  services.end(update);
  const auto read=mode=="content"?n.initial:mode=="old"?n.old:n.proposal;
  WeightedDescribeInput x;x.mass=n.mass;x.content=n.content;int64_t summary=0;
  auto& mem=services.begin();
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
  services.end(mem);
  auto& describe=services.begin();
  n.describe=append_weighted_describe(describe,x,n.input.bank,bind_weighted_describe(n.banks,n.input.payload.size(2)-1,summary,n.binding.regional),
    n.binding.regional,eps,error,remaining(describe));services.end(describe);
}
void append_weighted_node_full(TaskServices& services,WeightedFlowNode& n,const at::Tensor& error,const WeightedFlowBudget& remaining){
  auto& full=services.begin();
  if(auto c=std::get_if<weighted::KVConfig>(&n.binding.memory)){
    n.kv_full=append_weighted_kv_full(full,*c,n.proposal,n.input.times,n.content,n.input.bank,n.active,bind_weighted_kv_full(n.banks),error,remaining(full));n.full=n.kv_full->output;
  }else{
    n.recurrent_full=append_weighted_recurrent(full,std::get<weighted::RecurrentConfig>(n.binding.memory),WeightedRecurrentAction::Full,
      n.proposal,n.input.times,n.mass,n.content,n.input.bank,n.active,n.recurrent_parameters,error,remaining(full));n.full=n.recurrent_full->fields.at("G");
  }
  services.end(full);
}
void WeightedFlow::Impl::append_nodes(DeviceProgram& p,const ReadyBatch& ready){
  std::vector<WeightedNodeInput> inputs;
  for(size_t v=0;v<nodes.size();++v)inputs.push_back(append_weighted_node_input(p,ready,input_classes[v],v,boundary.batch_size,graph.inputs.size(),error));
  peers->append_local_send(p,graph,nodes,inputs,views,error,[&](DeviceProgram& worker){return remaining(worker);});
  for(size_t v=0;v<nodes.size();++v)if(node_devices[v]==device){auto& n=*nodes[v];n.input=inputs[v];
    append_weighted_node_local(p,*services,n,graph.regions[graph.nodes[v].region].read_mode,error,[&](DeviceProgram& worker){return remaining(worker);});
    views[v]=weighted_flow_view(n);
  }
  peers->append_local_receive(p,error);
}
void WeightedFlow::Impl::append_regions(DeviceProgram& p){
  for(auto& r:regions)append_weighted_region_views(p,*services,*r,views,error,[&](DeviceProgram& worker){return remaining(worker);});
}
ActionBatch WeightedFlow::Impl::append_full(DeviceProgram& p){
  const auto samples=boundary.batch_size,count=static_cast<int64_t>(nodes.size()),width=model.width();
  const auto opts=at::TensorOptions().dtype(model.nodes.at(0).bias.scalar_type()).device(device);
  ActionBatch out{at::zeros({count*samples,4},opts.dtype(at::kLong)),at::zeros({count*samples,width},opts),at::zeros({count*samples},opts.dtype(at::kBool))};
  for(int64_t v=0;v<count;++v)if(node_devices[v]==device){auto& n=*nodes[v];auto& view=views[v];n.control=view.control;n.active=view.active;
    append_weighted_node_full(*services,n,error,[&](DeviceProgram& worker){return remaining(worker);});
    n.payload=append_weighted_payload(p,n.control,n.full,n.active,out.values.scalar_type(),error);view.full=n.full;view.payload=n.payload;
  }
  peers->append_full(p,nodes,views,error,[&](DeviceProgram& worker){return remaining(worker);});
  for(int64_t v=0;v<count;++v){const auto& view=views[v];
    auto meta=out.coordinates.narrow(0,v*samples,samples);weighted_copy(p,meta.select(1,0),at::arange(samples,meta.options()));weighted_copy(p,meta.select(1,1),at::full({samples},v,meta.options()));
    weighted_copy(p,meta.select(1,2),view.times);weighted_copy(p,meta.select(1,3),view.times);
    weighted_copy(p,out.values.narrow(0,v*samples,samples),view.payload);weighted_copy(p,out.valid.narrow(0,v*samples,samples),view.active);
  }
  return out;
}
}
