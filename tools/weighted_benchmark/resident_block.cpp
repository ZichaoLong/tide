#include "resident_block.h"
#include "block_inputs.h"
#include "tide/resident_blocks.h"
#include "tide/parameters.h"
#include "portable_torch/runtime.hpp"
#include <ATen/core/grad_mode.h>
#include <cmath>
#include <set>
namespace weighted_benchmark {
using namespace tide::device_online;
namespace {
Json actual_work(const BlockWindow& w){
  std::vector<Tensor> counts;auto zero=at::zeros({},w.fields.at("output.strength").present.options().dtype(at::kLong));
  auto candidates=zero.clone(),fulls=zero.clone(),received=zero.clone();Index regions=0;
  for(const auto& [name,field]:w.fields)if(name.size()>9&&name.substr(name.size()-9)==".received"){
    const auto prefix=name.substr(0,name.size()-8);++regions;candidates=candidates+field.present.sum();
    fulls=fulls+w.fields.at(prefix+"message.strength").present.sum();
    received=received+w.metadata.at(prefix+"fiber_lengths").sum();
  }
  auto counts_cpu=at::stack({candidates,fulls,received,w.fields.at("edge.strength").present.sum(),
                            w.fields.at("output.strength").present.sum()}).cpu();
  const auto* x=counts_cpu.const_data_ptr<Index>();
  Json stats={{"candidate_events",x[0]},{"selected_events",x[1]},{"received_atoms",x[2]},
    {"visited_edges",x[3]},{"resident_regions",regions}};
  return {{"candidate_updates",x[0]},{"full_calls",x[1]},{"messages",x[3]},{"outputs",x[4]},
    {"pending",0},{"actual_stats",stats},{"input_positions",w.metadata.at("positions").numel()},
    {"boundary_activity",{{"scope","actual block presence/fiber counters; complete Settle cut"}}}};
}
class BlockCandidate final:public Resident {
  Fixture data;SettleGraph spec;at::Device device;bool training;std::string returns;Json selected;
  BlockResidentLimits limits;BlockInputs input;double scale;std::unique_ptr<BlockResidentSession> owner;
  ResidentParameterGradient gradient;
  Continuation initial()const{Continuation q;q.identity=data.graph.identity;q.batch_size=data.samples;return q;}
 public:
  BlockCandidate(const Json& j,Fixture f,BlockSelection selection,at::Device d,bool train,std::string output)
      :data(std::move(f)),spec(std::move(*selection.graph)),device(d),training(train),returns(std::move(output)),
       selected(std::move(selection.record)),input(block_inputs(spec,data)),scale(j.value("__task",Json::object()).value("loss_scale",.001)){
    at::NoGradGuard guard;const auto resolved=resident_block_limits(j,data);
#define FIELD(name) limits.name=resolved.at(#name).get<decltype(limits.name)>()
    FIELD(cache);FIELD(forward_bytes);FIELD(backward_bytes);FIELD(optimizer_bytes);FIELD(total_bytes);
    FIELD(workspace_bytes);FIELD(windows);FIELD(timeout_ms);
#undef FIELD
    selected["requested_limits"]=j.value("__resident_block_limits",Json::object());selected["effective_limits"]=resolved;
    selected["readiness"]="device";selected["selection"]="device";selected["progression"]="device";
    selected["optimizer_storage_constructed"]=true;selected["training"]=training;
    selected["inference_scope"]="public block session advance/detach; manual tape and optimizer storage are retained";
    selected["backward"]=training?"explicit first-order block VJP":"none";
    selected["task_loss_computed"]=true;
    OptimizerGroup group;for(const auto& p:data.model.parameters().owners())group.parameters.push_back(p.canonical);
    const auto task=j.value("__task",Json::object());group.lr=task.value("lr",.001);
    group.momentum=task.value("momentum",.3);group.weight_decay=task.value("weight_decay",.01);
    owner=std::make_unique<BlockResidentSession>(spec,data.model,initial(),device,ResidentOptimizerKind::sgd,
      std::vector<OptimizerGroup>{group},limits);
  }
  Step step(Index index)override{
    at::NoGradGuard guard;const auto cut=owner->cut();
    if(index<0||cut%data.span||index!=cut/data.span)throw std::invalid_argument("block window index disagrees with complete cut");
    auto positions=input.positions+index*data.positions;
    // The public owner clones/transfers boundary payload and presence inside
    // advance. The fixture is packed only once, without any model execution.
    auto window=owner->advance(positions,input.payload,input.present);
    const auto& c=window.fields.at("output.strength");const auto& g=window.fields.at("output.content");
    auto objective=block_objective(c.values,g.values,c.present,scale,training);
    Result out;out.continuation.identity=data.graph.identity;out.continuation.batch_size=data.samples;out.continuation.cut=owner->cut();
    if(returns=="outputs")out.outputs=block_outputs(spec,data.positions,c.values,g.values,c.present,window.metadata.at("output.times"));
    auto observation=actual_work(window);
    const auto loss=objective.value.cpu().item<double>();if(!std::isfinite(loss))throw std::runtime_error("nonfinite block task loss");
    if(training){
      BlockRoots roots;roots.token=window.token;
      roots.fields["output.strength"]={objective.strength_root,c.present};
      roots.fields["output.content"]={objective.content_root,c.present};
      gradient=owner->backward({roots}).parameters;
      if(!at::isfinite(gradient.values).all().item<bool>())throw std::runtime_error("nonfinite block parameter gradient");
      owner->step();
    }else owner->detach();
    portable_torch::synchronize(device);return {std::move(out),loss,std::move(observation)};
  }
  Json execution()const override{return selected;}
  std::map<std::string,Tensor> parameters()const override{
    at::NoGradGuard guard;std::map<std::string,Tensor> out;if(!training)return out;
    const auto c=owner->checkpoint();for(const auto& p:data.model.parameters().owners())out[p.canonical]=c.parameters.at(p.canonical);return out;
  }
  TrainingState training_state()const override{
    at::NoGradGuard guard;TrainingState out;if(!training)return out;const auto c=owner->checkpoint();
    const auto registry=data.model.parameters();const auto all=data.model.parameters(false).owners();
    for(const auto& p:registry.owners()){out.gradients[p.canonical]=Tensor{};out.momentum[p.canonical]=Tensor{};}
    if(gradient.values.defined()){
      const auto values=gradient.values.cpu(),connected=gradient.connected.cpu();
      for(size_t j=0;j<gradient.names.size();++j)if(registry.contains(gradient.names[j])&&connected[j].item<bool>()){
        const auto name=registry.canonical_name(gradient.names[j]);const auto p=registry.value(name);
        out.gradients[name]=values.narrow(0,gradient.offsets[j],p.numel()).reshape(p.sizes()).clone();}
    }
    std::set<std::string> momentum;
    for(const auto& group:c.groups)if(group.momentum!=0)momentum.insert(group.parameters.begin(),group.parameters.end());
    Index offset=0;for(size_t j=0;j<all.size();++j){const auto& p=all[j];
      if(registry.contains(p.canonical)&&momentum.count(p.canonical)&&c.slots.steps[j].item<Index>())
        out.momentum[p.canonical]=c.slots.first.narrow(0,offset,p.value.numel()).reshape(p.value.sizes()).clone();
      offset+=p.value.numel();}
    return out;
  }
  Continuation checkpoint()const override{at::NoGradGuard guard;return owner->continuation();}
  Index cut()const override{at::NoGradGuard guard;return owner->cut();}
  void save(const std::filesystem::path& path,int64_t max_bytes)const override{
    at::NoGradGuard guard;save_block_checkpoint(owner->checkpoint(),path.string(),max_bytes);
  }
  void restore(const std::filesystem::path& path,int64_t max_bytes)override{
    at::NoGradGuard guard;auto saved=load_block_checkpoint(path.string(),max_bytes);
    owner->close();owner.reset();owner=std::make_unique<BlockResidentSession>(spec,data.model,initial(),saved,device,limits);
    gradient={};
  }
  void close()override{at::NoGradGuard guard;if(owner)owner->close();}
};
}
std::unique_ptr<Resident> resident_block(const Json& j,Fixture data,BlockSelection selection,at::Device d,bool training,const std::string& returns){
  return std::make_unique<BlockCandidate>(j,std::move(data),std::move(selection),d,training,returns);
}
}
