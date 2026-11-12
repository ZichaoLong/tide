// Public installed Model -> block lowering -> actual device owner. Independent
// scalar Streaming scheduling checks integration; block_settle_formula owns the
// separate literal-mathematics anchor. No candidate data enters the reference.
#include "block_resident_fixture.h"
#include "block_resident_check_helpers.h"
#include "training_parameters.h"
#include "portable_torch/runtime.hpp"
#include <torch/csrc/autograd/autograd.h>
#include <ATen/Parallel.h>
#include <filesystem>
#include <unistd.h>
namespace {
using namespace tide;using namespace tide::device_online;using namespace tide::device_online::block_public_check;
constexpr Index B=2,L=3,P=2;
struct Input {Tensor positions,payload,present;std::vector<External> external;std::vector<Tensor> leaves;std::vector<Index> rows;};
Input input(const SettleGraph& g,const Continuation& q,int window){
  const Index offset=window*5;Input x;x.positions=at::tensor(std::vector<Index>{1+offset,2+offset,4+offset,1+offset,3+offset,4+offset},at::kLong).reshape({B,L});
  x.payload=at::full({B,L,P,3},.12,at::kFloat);x.payload.select(3,0).fill_(.3);x.payload[0][1][1][0]=0;
  x.present=at::ones({B,L,P},at::kBool);x.present[1][1][0]=false;
  auto counts=q.ledger;
  for(Index b=0;b<B;++b)for(Index l=0;l<L;++l)for(Index p=0;p<P;++p)if(x.present[b][l][p].item<bool>()){
    auto value=x.payload[b][l][p].clone().set_requires_grad(true);const auto key=Owner{b,p};
    auto occurrence=counts.count(key)?counts.at(key).first+1:0;const auto time=x.positions[b][l].item<Index>()*g.stride()+g.rank(g.graph().inputs[p]);
    counts[key]={occurrence,time};x.external.push_back({b,p,occurrence,time,value});x.leaves.push_back(value);x.rows.push_back((b*L+l)*P+p);}
  return x;
}
void output(const BlockWindow& actual,const Input& x,const Result& expected,const SettleGraph& graph){
  std::map<std::tuple<Index,Index,Index>,Tensor> rows;for(const auto& y:expected.outputs)rows[{y.batch,y.time,y.port}]=y.value;
  const auto mass=actual.fields.at("output.strength"),content=actual.fields.at("output.content");
  for(Index b=0;b<B;++b)for(Index l=0;l<L;++l)for(size_t p=0;p<graph.graph().outputs.size();++p){
    const auto time=x.positions[b][l].item<Index>()*graph.stride()+graph.rank(graph.graph().outputs[p]);const auto found=rows.find({b,time,p});const auto frame=b*L+l;
    require(mass.present[frame][p].cpu().item<bool>()==(found!=rows.end()),"public physical output presence differs");
    if(found!=rows.end()){close(mass.values[frame][p],found->second[0],"output c");close(content.values[frame][p],found->second.slice(0,1),"output G");}}
}
BlockRoots root(const BlockWindow& window){BlockRoots out;out.token=window.token;
  for(const auto& [name,value]:std::vector<std::pair<std::string,float>>{{"output.strength",.07f},{"output.content",.13f}}){
    const auto& f=window.fields.at(name);out.fields[name]={at::full_like(f.values,value),f.present};}return out;}
std::unique_ptr<BlockResidentSession> owner(const Fixture& f,at::Device device,const OptimizerGroup& group,const BlockCheckpoint* c=nullptr){
  BlockResidentLimits l;l.cache=8;l.forward_bytes=1LL<<29;l.backward_bytes=1LL<<30;l.total_bytes=3LL<<30;
  if(c)return std::make_unique<BlockResidentSession>(f.graph,f.model,f.initial,*c,device,l);
  return std::make_unique<BlockResidentSession>(f.graph,f.model,f.initial,device,ResidentOptimizerKind::adamw,std::vector<OptimizerGroup>{group},l);
}
void check(at::Device device,const std::string& path,bool restore){
  at::NoGradGuard guard;auto consumer=fixture();std::vector<training_detail::Version> unused;
  auto reference=training_detail::freeze_model(consumer.model,at::kCPU,unused);auto q=training_detail::freeze_continuation(consumer.initial);
  auto parameters=reference.parameters();OptimizerGroup group;group.lr=.003;group.eps=.1;group.weight_decay=.02;
  for(const auto& p:parameters.owners())group.parameters.push_back(p.canonical);AdamW optimizer(parameters,{group});
  std::optional<BlockCheckpoint> saved;if(restore)saved=load_block_checkpoint(path);
  auto session=owner(consumer,device,group,saved?&*saved:nullptr);std::unique_ptr<BlockResidentSession> resumed;
  for(int step=0;step<3;++step){std::vector<Input> inputs;std::vector<Result> reference_windows;std::vector<BlockRoots> bars;
    std::vector<BlockRoots> restored_bars;std::vector<Tensor> leaves,values,cotangents;const auto owners=parameters.owners();
    for(const auto& p:owners)leaves.push_back(p.value);
    for(int w=0;w<2;++w){inputs.push_back(input(consumer.graph,q,step*2+w));const auto& x=inputs.back();
      {at::AutoGradMode yes(true);Streaming executor(consumer.graph.graph(),reference,{});
        const auto stop=(x.positions.max().item<Index>()+1)*consumer.graph.stride();reference_windows.push_back(executor.run(q,x.external,stop,stop));q=reference_windows.back().continuation;}
      leaves.insert(leaves.end(),x.leaves.begin(),x.leaves.end());
      for(const auto& y:reference_windows.back().outputs){values.push_back(y.value);cotangents.push_back(at::tensor({.07f,.13f,.13f}));}
      if(restore&&step==0)continue;
      const auto actual=session->advance(x.positions,x.payload,x.present);output(actual,x,reference_windows.back(),consumer.graph);bars.push_back(root(actual));
      if(resumed)restored_bars.push_back(root(resumed->advance(x.positions,x.payload,x.present)));
    }
    std::vector<Tensor> expected;{at::AutoGradMode yes(true);expected=torch::autograd::grad(values,leaves,cotangents,false,false,true);}
    if(!(restore&&step==0)){const auto actual=session->backward(bars);std::map<std::string,size_t> names;
      for(size_t j=0;j<actual.parameters.names.size();++j)names[actual.parameters.names[j]]=j;
      for(size_t j=0;j<owners.size();++j){const auto index=names.at(owners[j].canonical);
        require(actual.parameters.connected[index].cpu().item<bool>()==expected[j].defined(),"public canonical None/zero differs "+owners[j].canonical);
        if(expected[j].defined())close(actual.parameters.values.narrow(0,actual.parameters.offsets[index],owners[j].value.numel()).reshape(owners[j].value.sizes()),expected[j],owners[j].canonical);}
      size_t k=owners.size();for(size_t w=0;w<inputs.size();++w)for(auto row:inputs[w].rows){
        const auto flags=actual.inputs[w].connected.reshape({-1});const auto gradients=actual.inputs[w].values.reshape({B*L*P,3});
        require(flags[row].cpu().item<bool>()==expected[k].defined(),"public input None/zero differs");if(expected[k].defined())close(gradients[row],expected[k],"input");++k;}
      session->step();if(resumed){resumed->backward(restored_bars);resumed->step();exact(session->checkpoint(),resumed->checkpoint());}
    }
    for(size_t j=0;j<owners.size();++j)owners[j].value.mutable_grad()=expected[j];optimizer.step();optimizer.zero_grad();q=training_detail::freeze_continuation(q);
    if(restore&&step==0)continue;
    continuation(session->continuation(),q);const auto c=session->checkpoint();
    for(const auto& p:parameters.owners())close(c.parameters.at(p.canonical),p.value,"updated "+p.canonical);
    if(step==0){save_block_checkpoint(c,path);auto loaded=load_block_checkpoint(path);exact(c,loaded);resumed=owner(consumer,device,group,&loaded);}
  }
  if(restore)exact(session->checkpoint(),load_block_checkpoint(path+".expected"));else save_block_checkpoint(session->checkpoint(),path+".expected");
  session->close();if(resumed)resumed->close();
  auto guarded=owner(consumer,device,group);const auto first=input(consumer.graph,consumer.initial,0);
  auto observation=guarded->advance(first.positions,first.payload,first.present);
  // This validity mask is derived in the public adapter, not part of the
  // internal sequence tape. It must still be guarded before roots are accepted.
  observation.fields.at("region.0.final.keys").valid.logical_not_();bool refused=false;
  try{guarded->backward({root(observation)});}catch(const std::logic_error&){refused=true;}
  require(refused,"public derived observation mask was not version guarded");guarded->close();
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{std::string path;bool restore=false;std::vector<char*> args{argv[0]};for(int j=1;j<argc;++j){const std::string x=argv[j];
      if(x=="--checkpoint-path"){if(++j==argc)throw std::invalid_argument("missing checkpoint path");path=argv[j];}
      else if(x=="--restore-check")restore=true;else args.push_back(argv[j]);}
    const bool temporary=path.empty();if(temporary)path="/tmp/tide-public-block-"+std::to_string(::getpid())+".pt";
    const auto options=portable_torch::parse_cli(args.size(),args.data(),true);if(options.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(options.device_spec=="auto"||options.dtype!=at::kFloat)throw std::invalid_argument("public block check requires explicit accelerator FP32");
    at::set_num_threads(1);at::set_num_interop_threads(1);check(portable_torch::resolve_device(options),path,restore);
    if(temporary){std::filesystem::remove(path);std::filesystem::remove(path+".expected");}
    std::cout<<"block-resident-public: passed actual_Model=true source_factors=true source_hidden_rank_padding=true independent_streaming=true AdamW_steps=3 windows=2 raw_continuation=true fresh_process="<<restore<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
