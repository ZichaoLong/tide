// Nonempty generic History fields are not covered by weighted singleton history.
#include "tide/graph_checkpoint.h"
#include "tide/region.h"
#include "tide/stream.h"
#include "tide/kernel.h"
#include "../src/graph_checkpoint_internal.h"
#include <stdexcept>
namespace {
using namespace tide;
class RecordHistory final:public RegionKernel {
 public:
  History initial(const RegionWeights&,const RegionLayout&,const Tensor& ref)const override{
    return {-1,{{"signed",-7}},{{"seen",{{0,(Index{1}<<54)+3}}}},{{"memory",at::full({},.25,ref.options())}}};
  }
  Selection step(const RegionWeights&,const RegionInput& r)const override{
    Selection out;out.history=r.history;out.history.last_time=r.time;
    out.history.scalars["signed"]-=2;out.history.node_maps["seen"][0]+=1;
    for(const auto& c:r.candidates){out.active.insert(c.node);out.controls[c.node]=at::ones({},r.payload_options);
      out.history.tensors["memory"]=out.history.tensors.at("memory")+c.descriptor;}
    return out;
  }
  void validate_weights(const RegionWeights&,const RegionLayout&)const override{}
  void validate_history(const History& h,const RegionLayout&)const override{
    if(h.scalars.size()!=1||!h.scalars.count("signed")||h.node_maps.size()!=1||!h.node_maps.count("seen")
        ||h.tensors.size()!=1||!h.tensors.count("memory"))throw std::invalid_argument("record history domain");
  }
};
}
void check_native_graph_records(const std::filesystem::path& directory,at::Device device,at::ScalarType dtype){
  using namespace tide;auto o=at::TensorOptions().dtype(dtype).device(device);
  Graph g;g.nodes={{0}};g.inputs=g.outputs={0};g.regions={{1,true,false,"proposal","record-history-check"}};g.compile();
  Model m;m.nodes={{at::zeros({1},o),at::ones({1,1},o),at::zeros({1},o),at::ones({1},o)}};
  m.regions={{{},std::make_shared<RecordHistory>()}};m.input_scale=m.output_scale={at::ones({},o)};configure_model(g,m);
  Continuation q;q.identity=g.identity;
  q.states[{0,0}]=m.nodes[0].kernel->initial(m.nodes[0]);
  q.history[{0,0}]=m.regions[0].kernel->initial(m.regions[0],region_layout(g,0),m.nodes[0].bias);
  auto initial_file=directory/"initial-graph.tidegraph";GraphCheckpoint::save(initial_file,g,m,q);
  auto initial=GraphCheckpoint::load(initial_file,g,m);
  if(initial.states.at({0,0}).last_time!=-1||initial.history.at({0,0}).last_time!=-1)
    throw std::runtime_error("native graph checkpoint lost initial negative clock");
  auto result=Streaming(g,m,Options{}).run(q,{{0,0,0,1,at::ones({1},o)}},3,3);
  auto file=directory/"generic-history.tidegraph";GraphCheckpoint::save(file,g,m,result.continuation);
  const auto size=static_cast<int64_t>(std::filesystem::file_size(file));
  auto bounded=directory/"bounded-graph.tidegraph";
  GraphCheckpoint::save(bounded,g,m,result.continuation,nullptr,size);
  if(checkpoint_detail::read_all(bounded)!=checkpoint_detail::read_all(file))
    throw std::runtime_error("checkpoint budget changed serialized values");
  auto refused=directory/"refused-graph.tidegraph";bool over=false;
  try{GraphCheckpoint::save(refused,g,m,result.continuation,nullptr,size-1);}catch(const std::invalid_argument&){over=true;}
  if(!over||std::filesystem::exists(refused))throw std::runtime_error("graph save ignored file byte limit");
  over=false;auto before=m.nodes[0].weight.clone();
  try{GraphCheckpoint::load(file,g,m,nullptr,size-1);}catch(const std::invalid_argument&){over=true;}
  if(!over||!at::equal(before,m.nodes[0].weight))throw std::runtime_error("graph load ignored budget or mutated owner");
  GraphCheckpoint::load(file,g,m,nullptr,size);
  auto actual=GraphCheckpoint::load(file,g,m);const auto& a=actual.history.at({0,0});const auto& b=result.continuation.history.at({0,0});
  if(a.last_time!=b.last_time||a.scalars!=b.scalars||a.node_maps!=b.node_maps||!at::equal(a.tensors.at("memory"),b.tensors.at("memory")))
    throw std::runtime_error("native graph checkpoint lost signed/nonempty history");
  // A structurally invalid record with a valid checksum must fail before even
  // the fixed model values are changed. A save-side rejection is not this test.
  using namespace tide::checkpoint_detail;
  auto bad=decode_graph(read_all(file));bad.continuation.states.at({0,0}).last_time=bad.continuation.cut;
  for(auto& owner:bad.parameters.owners)owner.value=owner.value+3;
  auto invalid=directory/"invalid-clock.tidegraph";publish_exclusive(invalid,encode_graph(bad));
  auto registry=m.parameters(false);std::map<std::string,Tensor> previous;
  for(const auto& owner:registry.owners())previous.emplace(owner.canonical,owner.value.clone());
  bool rejected=false;try{GraphCheckpoint::load(invalid,g,m);}catch(const std::invalid_argument&){rejected=true;}
  if(!rejected)throw std::runtime_error("native graph checkpoint accepted invalid loaded clock");
  for(const auto& owner:registry.owners())if(!at::equal(previous.at(owner.canonical),owner.value))
    throw std::runtime_error("native graph checkpoint mutated parameters before complete validation");
}
