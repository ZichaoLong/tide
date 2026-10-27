// Capacity refusal, exact disk codec and unbounded cumulative logical time.
#include "weighted_flow.h"
#include "weighted_flow_internal.h"
#include "../../cpp/test/weighted_graph_fixture.h"
#include "../../cpp/bench/streaming.h"
#include "tide/graph_checkpoint.h"
#include "tide/greedy.h"
#include "portable_torch/runtime.hpp"
#include <cmath>
#include <unistd.h>
namespace tide::device_online {
namespace {
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
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
}
void check_weighted_flow_boundaries(at::Device device,at::ScalarType dtype,const std::filesystem::path& directory){
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
  Greedy reference(g,m,Options{});compare(result,reference.run(snapshot,tail,10,10),dtype);
  // No pending in this DAG. Crossing a huge empty time interval must not use
  // one stage/tick or reinterpret original time as a reception counter.
  const Index time=(Index{1}<<54)+31;std::vector<External> later;
  for(Index b=0;b<2;++b)for(Index port=0;port<2;++port)later.push_back({b,port,2,time,at::tensor({.6,.2,-.3},at::kFloat).to(dtype)});
  auto distant=restored.advance(later,time+4);compare(distant,reference.run(result.continuation,later,time+4,time+4),dtype);
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
  compare(Result{retry.checkpoint()},Result{q},dtype,true);compare(retry.advance(xs,10),reference.run(q,xs,10,10),dtype);retry.close();
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
