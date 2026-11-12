// CPU-only adapter checks: actual static bindings and independent scalar loss.
// This does not instantiate a resident owner or certify any device kernel.
#include "resident_policy.h"
#include "block_inputs.h"
#include "tide/parameters.h"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <iostream>
#include <limits>
using namespace weighted_benchmark;
namespace {
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
template<class F>void rejects(F fn){try{fn();}catch(const std::exception&){return;}throw std::runtime_error("invalid block adapter request accepted");}
Fixture material(){
  Fixture f;f.samples=2;f.positions=3;f.span=15;
  f.graph.nodes={Node{0},Node{0},Node{1}};f.graph.regions={Region{2},Region{1}};
  f.graph.edges={{0,2,2},{0,2,2},{1,2,2}};f.graph.inputs={0,0,1};f.graph.outputs={2,2,1};f.graph.compile();
  weighted::KVConfig c;c.width=2;c.summary_width=2;c.adapter_rank=1;c.ffn_width=3;
  weighted::RegionalConfig r;SourceClasses classes{{0,0,0},{0,0,0}};
  for(size_t v=0;v<f.graph.nodes.size();++v){
    auto x=at::zeros({1},at::kFloat);NodeWeights n{x.clone(),x.clone(),x.clone(),x.clone()};
    c.memory_read=v==2?"qry":"sum";weighted::configure_node(n,c,r,classes,0);f.model.nodes.push_back(std::move(n));}
  for(size_t rindex=0;rindex<2;++rindex){RegionWeights w;weighted::configure_region(w,r);f.model.regions.push_back(std::move(w));}
  const auto value=at::tensor({.0f,.4f,-.2f});
  // Out-of-order records, two physical ports sharing one material tensor, a
  // zero-strength present message, missing slots, and noncontiguous positions.
  f.inputs={{0,0,1,11,value},{1,2,0,6,at::tensor({.5f,.1f,.7f})},
            {0,1,0,1,value},{0,0,0,1,value}};return f;
}
}
int main(){try{
  at::set_num_threads(1);const auto f=material();const auto target=at::Device("cuda:0");
  Json j={{"__resident_executor","auto"},{"workload",{{"stride",5},{"graph",{{"ranks",{1,3}}}}}}};
  auto select=[&](const Json& data,const Fixture& fixture,bool prefill=true,at::ScalarType dtype=at::kFloat,
                  const std::string& mode="outputs",std::vector<at::Device> peers={}){
    return select_resident_block(data,fixture,prefill,dtype,target,peers,{},mode);};
  auto selected=select(j,f);require(selected.graph.has_value()&&selected.record.at("effective")=="block-v2","KV sum/query auto selection failed");
  require(!select(j,f,true,at::kFloat,"complete").graph,"complete observations silently selected block");
  require(!select(j,f,false).graph&&!select(j,f,true,at::kHalf).graph,"streaming/half silently selected block");
  require(!select(j,f,true,at::kFloat,"outputs",{target,at::Device("cuda:1")}).graph,"multiple owners selected block");
  auto old=j;old.erase("__resident_executor");require(!select(old,f).graph,"historical fixture silently adopted block");
  auto strict=j;strict["__resident_executor"]="block-v2";
  rejects([&]{select(strict,f,true,at::kFloat,"complete");});
  auto records=j;records["__input_interface"]="records";require(!select(records,f).graph,"explicit records lost");
  auto limits=j;limits["__resident_general_limits"]=true;require(!select(limits,f).graph,"old budgets changed owner");
  limits["__resident_block_limits"]=Json::object();rejects([&]{select(limits,f);});
  auto no_certificate=j;no_certificate.erase("workload");require(!select(no_certificate,f).graph,"guessed Settle certificate");
  auto incompatible=f;auto c=std::get<weighted::KVConfig>(weighted::node_binding(incompatible.model.nodes[1]).memory);
  c.memory_read="qry";weighted::RegionalConfig r;weighted::configure_node(incompatible.model.nodes[1],c,r,SourceClasses{{0,0,0},{0,0,0}},0);
  require(!select(j,incompatible).graph,"heterogeneous in-region geometry selected block");
  auto recurrent=f;weighted::configure_node(recurrent.model.nodes[2],weighted::RecurrentConfig{},r,SourceClasses{{0,0,0},{0,0,0}},0);
  require(!select(j,recurrent).graph,"unimplemented resident recurrent selected block");
  auto budget=j;budget["__logical_windows"]=30;auto resolved=resident_block_limits(budget,f);
  require(resolved.at("cache")==90,"logical raw bound lost");
  budget["__resident_block_limits"]={{"cache",19},{"timeout_ms",77}};resolved=resident_block_limits(budget,f);
  require(resolved.at("cache")==19&&resolved.at("timeout_ms")==77,"explicit block budget substituted");
  budget["__resident_block_limits"]={{"cache",true}};rejects([&]{resident_block_limits(budget,f);});
  budget["__resident_block_limits"]={{"queue",5}};rejects([&]{resident_block_limits(budget,f);});
  budget["__resident_block_limits"]={{"timeout_ms",int64_t(std::numeric_limits<int32_t>::max())+1}};
  rejects([&]{resident_block_limits(budget,f);});
  auto resumed=j;resumed["__resident_block_resume_limits"]=resolved;resumed["__logical_windows"]=300;
  require(resident_block_limits(resumed,f)==resolved,"resume changed bound raw shape");
  const auto packed=block_inputs(*selected.graph,f);
  require(packed.positions.scalar_type()==at::kLong&&packed.payload.sizes()==at::IntArrayRef{2,3,3,3},"input tensor layout differs");
  require(packed.present.sum().item<Index>()==4&&packed.present[0][0][0].item<bool>()&&
    !packed.present[0][1][0].item<bool>()&&packed.payload[0][0][0][0].item<float>()==0,"presence/zero conflated");
  require(at::equal(packed.payload[0][0][0],packed.payload[0][0][1]),"physical shared input value differs");
  auto invalid=f;invalid.inputs.push_back(invalid.inputs.back());rejects([&]{block_inputs(*selected.graph,invalid);});
  invalid=f;invalid.inputs.back().position=7;rejects([&]{block_inputs(*selected.graph,invalid);});
  // Independent scalar objective with autograd versus the two analytic field
  // roots, including a connected zero strength. Absent NaNs cannot seed roots.
  auto strength=at::tensor({0.f,.2f,-.3f,.4f,.7f,.1f}).reshape({2,3}).set_requires_grad(true);
  auto content=at::arange(12,at::kFloat).reshape({2,3,2}).div(7).set_requires_grad(true);
  auto present=at::tensor({1,1,0,1,0,1},at::kLong).to(at::kBool).reshape({2,3});
  auto expected=at::zeros({});for(Index b=0;b<2;++b)for(Index p=0;p<3;++p)if(present[b][p].item<bool>())
    for(Index d=0;d<2;++d)expected=expected+.3*(strength[b][p]*content[b][p][d]).square();
  const auto bars=torch::autograd::grad({expected},{strength,content});
  const auto actual=block_objective(strength.detach(),content.detach(),present,.3,true);
  require(at::allclose(actual.value,expected)&&at::allclose(actual.strength_root,bars[0])&&at::allclose(actual.content_root,bars[1]),"physical output formula/VJP differs");
  const auto inference=block_objective(strength.detach(),content.detach(),present,.3,false);
  require(at::equal(inference.value,actual.value)&&!inference.strength_root.defined()&&!inference.content_root.defined(),"inference/metrics changed task or built gradient roots");
  auto absent_content=content.detach().clone();absent_content[0][2].fill_(std::numeric_limits<float>::quiet_NaN());
  const auto masked=block_objective(strength.detach(),absent_content,present,.3,true);
  require(at::isfinite(masked.value).item<bool>()&&at::isfinite(masked.content_root).all().item<bool>()&&at::isfinite(masked.strength_root).all().item<bool>(),"absent NaN leaked into objective/roots");
  auto times=at::tensor({8,8,6,3,3,1},at::kLong).reshape({2,3});
  const auto outputs=block_outputs(*selected.graph,1,strength.detach(),content.detach(),present,times);
  require(outputs.size()==4&&outputs.front().time==1&&outputs[2].port==0&&outputs[3].port==1,"physical output ordering/duplicates lost");
  std::cout<<"standalone block policy, inputs and independent task VJP passed (CPU adapter only)\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
