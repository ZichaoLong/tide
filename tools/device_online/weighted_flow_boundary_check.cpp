// Capacity refusal, exact disk codec and unbounded cumulative logical time.
#include "weighted_flow.h"
#include "weighted_precision_reference.h"
#include "weighted_flow_internal.h"
#include "../../cpp/test/weighted_graph_fixture.h"
#include "../../cpp/bench/streaming.h"
#include "tide/graph_checkpoint.h"
#include "tide/greedy.h"
#include "tide/allocation_failure.h"
#include "portable_torch/runtime.hpp"
#include <cmath>
#include <unistd.h>
namespace tide::device_online {
namespace {
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void fiber_capacity(){
  Graph g;g.nodes.resize(4);g.inputs={0,0,2};
  g.edges={{0,1,1},{0,1,1},{1,1,2}}; // Parallel identities and feedback count separately.
  require(weighted_node_fiber_capacity(g,0,4096)==2&&weighted_node_fiber_capacity(g,1,4096)==3&&
    weighted_node_fiber_capacity(g,2,4096)==1&&weighted_node_fiber_capacity(g,3,4096)==1,
    "physical fiber bound omitted a port/parallel edge or empty row");
  require(weighted_node_fiber_capacity(g,1,2)==2,"global queue bound not retained");
}
void compare(const Result& a,const Result& b,at::ScalarType dtype,bool exact=false){tide_bench::compare(a,b,true,dtype,std::nullopt,exact?0.:dtype==at::kHalf?2e-2:-1.,exact?0.:dtype==at::kHalf?1e-3:-1.);}
void half_payload(at::Device device){
  at::NoGradGuard no_grad;auto f=at::TensorOptions().device(device).dtype(at::kFloat);
  auto control=at::full({2},0x1p-24,f),full=at::full({2,1},0x1p-25,f),active=at::zeros({2},f.dtype(at::kBool)),error=at::zeros({1},f.dtype(at::kInt));
  active[0]=true;control[1]=NAN;full[1].fill_(NAN);
  DeviceProgram p(device);auto payload=append_weighted_payload(p,control,full,active,at::kHalf,error);p.finish();portable_torch::synchronize(device);p.run();
  require(!error.cpu().item<int>()&&payload[0][0].cpu().item<float>()==0x1p-24&&payload[0][1].cpu().item<float>()==0,"half payload subnormal/content underflow or inactive poison differs");
  control[0]=0x1p-25;portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==24,"half sending support underflow not refused");
  error.zero_();control[0]=1;full[0].fill_(1e5);portable_torch::synchronize(device);p.run();require(error.cpu().item<int>()==24,"half Full overflow not refused");p.close();
}
void parameter_banks(at::Device device,at::ScalarType dtype){
  at::NoGradGuard no_grad;
  const std::string prefix="full_program.content.";
  std::map<std::string,at::Tensor> parameters;
  for(const auto& name:{"query","out"})parameters[prefix+name]=at::ones({2,2},at::TensorOptions().dtype(dtype));
  auto tied=at::ones({3,2},at::TensorOptions().dtype(dtype));
  parameters[prefix+"gate"]=parameters[prefix+"up"]=tied;parameters[prefix+"down"]=tied.t();
  constexpr int64_t bytes=104; // 2*(2*2*4) + 3*(3*2*4), including tied roles.
  WeightedBanks short_bank(parameters,"",device);bool refused=false;
  try{bind_weighted_kv_full(short_bank,bytes-1);}catch(const std::invalid_argument& e){
    refused=std::string(e.what()).find("before allocation")!=std::string::npos;}
  require(refused&&!short_bank.roles().count("full.down"),"parameter group exceeded budget before refusal");
  int64_t used=0;for(const auto& [_,role]:short_bank.roles())used+=role.bank.nbytes();
  require(used<=bytes-1,"over-budget parameter role allocated");
  WeightedBanks exact(parameters,"",device);auto full=bind_weighted_kv_full(exact,bytes);
  require(full.residual.gate.data_ptr()!=full.residual.up.data_ptr(),"tied parameter roles shared mutable bank storage");
  require(at::equal(full.residual.down.cpu(),tied.t().to(at::kFloat).unsqueeze(0)),"bank transpose/half conversion differs");
  auto pool=std::make_shared<ParameterBankPool>();
  WeightedBanks pooled(parameters,"",device,pool);auto shared=bind_weighted_kv_full(pooled,bytes-24);
  require(shared.residual.gate.data_ptr()==shared.residual.up.data_ptr(),"frozen pool did not reuse actual parameter alias");
  require(shared.query.data_ptr()!=shared.residual.out.data_ptr(),"frozen pool merged independent equal values");
  WeightedBanks second(parameters,"",device,pool);auto reused=bind_weighted_kv_full(second,0);
  require(reused.query.data_ptr()==shared.query.data_ptr(),"frozen pool charged/duplicated a previously bound slice");
  parameters[prefix+"gate"]=at::ones({1},at::TensorOptions().dtype(dtype)).expand({1000000,2});
  WeightedBanks large(parameters,"",device);refused=false;
  try{bind_weighted_kv_full(large,1024*1024);}catch(const std::invalid_argument& e){
    refused=std::string(e.what()).find("before allocation")!=std::string::npos;}
  require(refused&&!large.roles().count("full.gate"),"large bank materialized before admission");
}
void allocation_snapshot(at::Device device,at::ScalarType dtype){
  at::NoGradGuard no_grad;
  auto g=weighted_check::graph("fc","kv");auto m=weighted_check::model(g,"kv",at::kCPU,dtype);
  auto r=weighted::region_binding(m.regions[0]);r.regularizer="bern";r.gamma_max=20;r.k=1;
  for(auto& w:m.regions){w.extra.at("head.vector").zero_();w.extra.at("head.bias").zero_();weighted::configure_region(w,r);}
  for(size_t i=0;i<m.nodes.size();++i){auto& w=m.nodes[i];auto b=weighted::node_binding(w);
    w.extra.at("read_program.score.vector").zero_();w.extra.at("read_program.score.bias").fill_(i%2?-2000:2000);
    weighted::configure_node(w,std::get<weighted::KVConfig>(b.memory),r,b.sources,b.aggregate_alpha);}
  Continuation q;q.identity=g.identity;q.batch_size=2;auto xs=weighted_check::inputs(g,at::kCPU,dtype);
  for(auto& x:xs)x.value[0]=.05;
  WeightedFlowLimits limits;limits.cache=8;limits.stages=32;
  WeightedFlow flow(g,m,q,device,limits);bool caught=false;
  try{flow.advance(xs,10);}catch(const weighted::AllocationFailure& e){
    caught=e.details.at("solver")=="resident-fp32"&&e.details.at("error_code")=="21"&&e.time==1;
    require(e.nodes==std::vector<Index>({0,1}),"resident failure lost candidate node IDs");
    require(e.tensors.at("scores").scalar_type()==at::kFloat&&e.tensors.at("partial_controls").numel()==2,
      "resident failure did not capture actual row tensors");
    require(!e.tensors.count("controls"),"early underflow exposed a complete successful control row");
  }
  require(caught,"resident allocation failure lacked typed snapshot");
  bool rejected=false;try{flow.advance({},10);}catch(const std::logic_error&){rejected=true;}
  require(rejected,"failed resident allocation owner reused without restore");flow.close();
}
}
void check_weighted_flow_boundaries(at::Device device,at::ScalarType dtype,const std::filesystem::path& directory){
  fiber_capacity();
  parameter_banks(device,dtype);
  allocation_snapshot(device,dtype);
  if(dtype==at::kHalf)half_payload(device);
  auto g=weighted_check::graph("fc","kv");auto m=weighted_check::model(g,"kv",at::kCPU,dtype);
  Continuation q;q.identity=g.identity;q.batch_size=2;auto xs=weighted_check::inputs(g,at::kCPU,dtype);
  at::NoGradGuard no_grad;WeightedFlowLimits limits;limits.cache=8;limits.stages=32;
  WeightedFlow initial(g,m,q,device,limits);std::vector<External> head,tail;
  for(const auto& x:xs)(x.time<4?head:tail).push_back(x);
  const auto reserved=initial.reserved_bytes();auto first=initial.advance(head,4);auto snapshot=initial.checkpoint();initial.close();
  std::filesystem::create_directories(directory);const auto path=directory/("weighted-resident-"+std::to_string(getpid())+".pt");
  GraphCheckpoint::save(path,g,m,snapshot);auto other=weighted_check::model(g,"kv",at::kCPU,dtype);
  auto loaded=GraphCheckpoint::load(path,g,other);compare(Result{loaded},Result{snapshot},dtype,true);
  WeightedFlow restored(g,other,loaded,device,limits);auto result=restored.advance(tail,10);
  Greedy reference(g,dtype==at::kHalf?weighted_precision_check::reference_model(m):m,Options{});
  compare(result,reference.run(weighted_precision_check::reference_continuation(snapshot),weighted_precision_check::reference_inputs(tail),10,10),dtype);
  // No pending in this DAG. Crossing a huge empty time interval must not use
  // one stage/tick or reinterpret original time as a reception counter.
  const Index time=(Index{1}<<54)+31;std::vector<External> later;
  for(Index b=0;b<2;++b)for(Index port=0;port<2;++port)later.push_back({b,port,2,time,at::tensor({.6,.2,-.3},at::kFloat).to(dtype)});
  auto distant=restored.advance(later,time+4);compare(distant,reference.run(weighted_precision_check::reference_continuation(result.continuation),weighted_precision_check::reference_inputs(later),time+4,time+4),dtype);
  require(distant.stats.at("resident_stages")<limits.stages,"resident logical-time gap consumed stage capacity");restored.close();
  std::filesystem::remove(path);
  auto reject=[&](WeightedFlowLimits cap,const std::vector<External>& input,const char* marker){
    WeightedFlow owner(g,m,q,device,cap);bool refused=false;
    try{owner.advance(input,10);}catch(const std::exception& e){refused=std::string(e.what()).find(marker)!=std::string::npos;}
    require(refused,"weighted capacity refusal missing");owner.close();
  };
  auto cap=limits;cap.queue=2;reject(cap,{xs[0]},"device error=1");cap=limits;cap.stages=1;reject(cap,xs,"device error=4");
  cap=limits;cap.cache=1;reject(cap,xs,"device error=");
  WeightedFlow retry(g,m,q,device,limits);auto bad=xs;bad[0].position=9;bool refused=false;
  try{retry.advance(bad,10);}catch(const std::invalid_argument&){refused=true;}require(refused,"bad boundary accepted");
  compare(Result{retry.checkpoint()},Result{q},dtype,true);compare(retry.advance(xs,10),reference.run(q,weighted_precision_check::reference_inputs(xs),10,10),dtype);retry.close();
  cap=limits;cap.tensor_bytes=1;refused=false;
  try{WeightedFlow too_large(g,m,q,device,cap);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"weighted graph tensor budget omitted");
  cap=limits;cap.cache=(Index{1}<<50);refused=false;
  try{WeightedFlow huge(g,m,q,device,cap);}catch(const std::invalid_argument& e){refused=std::string(e.what()).find("before allocation")!=std::string::npos;}
  require(refused,"large raw-state budget not checked before allocation");
  cap=limits;cap.tensor_bytes=reserved-1;refused=false;
  try{WeightedFlow short_budget(g,m,q,device,cap);}catch(const std::invalid_argument&){refused=true;}
  require(refused,"complete graph tensor budget omitted retained storage");
}
}
