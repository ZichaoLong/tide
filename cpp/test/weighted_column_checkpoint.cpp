// Public standalone window owner vs independent Streaming: raw state, ports,
// VJP/three SGD steps and a real exec'd-process checkpoint continuation.
#include "weighted_graph_fixture.h"
#include "tide/weighted_column_checkpoint.h"
#include "tide/graph_checkpoint.h"
#include "tide/optimizer.h"
#include "tide/stream.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <filesystem>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace tide;
namespace col=weighted::columns;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void close(const Tensor& a,const Tensor& b,bool exact=false){
  require(a.defined()==b.defined(),"column owner disconnected/zero mismatch");if(!a.defined())return;
  require(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type(),"column owner tensor metadata mismatch");
  require(exact||!a.is_floating_point()?at::equal(a,b):at::allclose(a,b,1e-8,1e-10),"column owner numerical mismatch");
}
void state(const Continuation& a,const Continuation& b,bool exact=false){
  require(a.identity==b.identity&&a.cut==b.cut&&a.batch_size==b.batch_size&&a.ledger==b.ledger&&
    a.pending.empty()&&b.pending.empty()&&a.states.size()==b.states.size()&&a.history.size()==b.history.size(),"column owner cursor mismatch");
  for(const auto& [owner,s]:a.states){const auto& t=b.states.at(owner);
    require(s.last_time==t.last_time&&s.observations==t.observations&&s.slots.size()==t.slots.size(),"column owner raw clock mismatch");
    close(s.value,t.value,exact);for(const auto& [name,x]:s.slots)close(x,t.slots.at(name),exact);
  }
  for(const auto& [owner,h]:a.history){const auto& k=b.history.at(owner);
    require(h.last_time==k.last_time&&h.scalars==k.scalars&&h.node_maps==k.node_maps&&h.tensors.empty()&&k.tensors.empty(),"column owner history mismatch");}
}
Continuation detach(Continuation q){
  for(auto& [_,s]:q.states){s.value=s.value.detach();for(auto& [__,x]:s.slots)x=x.detach();}return q;
}
struct Input {col::Layout layout;std::map<Index,col::Message> ports;std::vector<External> records;Tensor values;Index stop;};
Input input(const SettleGraph& spec,Index step,const Continuation& q){
  const Index base=(Index{1}<<54)+step*5;Input out;
  out.layout={{0,2,4},{base,base+2,base+1,base+3}};out.stop=base+4;
  out.values=at::tensor({.4,.2,-.3, .0,.3,.1, .6,-.1,.5, .9,.4,-.2},at::kDouble).reshape({4,3}).set_requires_grad(true);
  // Independent port calendars, one zero-mass occurrence, and a silent sample
  // on later windows. A row remains present even when its message mass is zero.
  std::vector<std::vector<Index>> rows=step%2?std::vector<std::vector<Index>>{{0,1},{1}}:
                                                 std::vector<std::vector<Index>>{{0,1,2,3},{0,2}};
  for(Index p=0;p<2;++p){
    auto ids=at::tensor(rows[p],at::kLong);auto x=out.values.index_select(0,ids);
    out.ports[p]={rows[p],x.select(1,0),x.slice(1,1)};
    std::map<Index,Index> count;
    for(size_t i=0;i<rows[p].size();++i){auto row=rows[p][i];const Index b=row/2;
      const auto found=q.ledger.find({b,p});const auto start=found==q.ledger.end()?0:found->second.first+1;
      out.records.push_back({b,p,start+count[b]++,out.layout.times[row]*spec.stride()+spec.rank(spec.graph().inputs[p]),x[i]});
    }
  }
  return out;
}
Tensor loss(const col::Result& result){
  std::vector<Tensor> terms;for(const auto& [_,p]:result.outputs)if(!p.rows.empty())
    terms.push_back((p.strength.unsqueeze(1)*p.content).square().sum()*.001);
  require(!terms.empty(),"test has no column outputs");return at::stack(terms).sum();
}
Tensor loss(const tide::Result& result){
  std::vector<Tensor> terms;for(const auto& p:result.outputs)terms.push_back((p.value[0]*p.value.slice(0,1)).square().sum()*.001);
  require(!terms.empty(),"test has no scalar outputs");return at::stack(terms).sum();
}
Tensor row(const col::Result& result,const col::Message& p,Index batch,Index position){
  Index count=-1;for(size_t i=0;i<p.rows.size();++i){auto r=p.rows[i];
    if(r>=result.layout.offsets[batch]&&r<result.layout.offsets[batch+1]&&result.layout.times[r]==position)count=i;}
  require(count>=0,"missing actual column output/message");return at::cat({p.strength[count].reshape({1}),p.content[count]});
}
void observations(const SettleGraph& spec,const col::WindowResult& a,const tide::Result& b){
  state(a.continuation,b.continuation);Index output_count=0,message_count=0;
  for(const auto& [_,p]:a.columns.outputs)output_count+=p.rows.size();
  for(const auto& [_,p]:a.columns.messages)message_count+=p.rows.size();
  require(output_count==static_cast<Index>(b.outputs.size())&&message_count==static_cast<Index>(b.messages.size()),"column support count differs");
  for(const auto& p:b.outputs)close(row(a.columns,a.columns.outputs.at(p.port),p.batch,p.time/spec.stride()),p.value);
  for(const auto& p:b.messages)close(row(a.columns,a.columns.messages.at(p.source),p.batch,p.time/spec.stride()),p.value);
}
OptimizerGroup group(const ParameterRegistry& registry){
  OptimizerGroup g;g.lr=.001;g.momentum=.3;g.weight_decay=.01;
  for(const auto& p:registry.owners())g.parameters.push_back(p.canonical);return g;
}
void owners(const ParameterRegistry& a,const ParameterRegistry& b,const SGD& oa,const SGD& ob,bool exact=false){
  require(a.alias_partitions()==b.alias_partitions(),"column shared parameter aliases differ");
  const auto xs=a.owners(),ys=b.owners();require(xs.size()==ys.size(),"column parameter domains differ");
  for(size_t i=0;i<xs.size();++i){require(xs[i].canonical==ys[i].canonical,"column canonical owner differs");
    close(xs[i].value,ys[i].value,exact);close(xs[i].value.grad(),ys[i].value.grad(),exact);}
  require(oa.state().size()==ob.state().size(),"column optimizer state domain differs");
  for(const auto& [name,s]:oa.state())close(s.momentum_buffer,ob.state().at(name).momentum_buffer,exact);
}
struct TestOwner {
  Graph graph;Model model;SettleGraph spec;ParameterRegistry registry;SGD optimizer;col::WindowExecutor engine;Continuation q;
  TestOwner(const std::string& family,const std::string& topology="fc")
    :graph(weighted_check::graph(topology,family)),model(weighted_check::model(graph,family,at::kCPU,at::kDouble)),
     spec(graph,topology=="hb"?std::vector<Index>{1,2,2,3}:std::vector<Index>{1,2,3},"ports",4),
     registry(model.parameters()),optimizer(registry,{group(registry)}),engine(spec,model){q.identity=graph.identity;q.batch_size=2;}
  col::WindowResult step(Index index){
    optimizer.zero_grad();auto x=input(spec,index,q);auto result=engine.run_ports(q,x.layout,x.ports,x.stop);
    optimizer.backward(loss(result.columns));optimizer.step();q=detach(result.continuation);return result;
  }
  void save(const std::filesystem::path& p){GraphCheckpoint::save(p,graph,model,q,&optimizer);}
  void restore(const std::filesystem::path& p){q=GraphCheckpoint::load(p,graph,model,&optimizer);}
};
template<class F> void rejects(F operation){
  try{operation();}catch(const std::invalid_argument&){return;}throw std::runtime_error("column invalid cursor accepted");
}
void check(const std::string& family,const std::string& topology){
  TestOwner a(family,topology),b(family,topology);Streaming scalar(b.graph,b.model,Options{});Continuation expected=b.q;
  for(Index step=0;step<3;++step){
    auto x=input(a.spec,step,a.q);a.optimizer.zero_grad();b.optimizer.zero_grad();
    auto actual=a.engine.run_ports(a.q,x.layout,x.ports,x.stop);
    auto reference=scalar.run(expected,x.records,x.stop*a.spec.stride(),x.stop*a.spec.stride());
    observations(a.spec,actual,reference);close(loss(actual.columns),loss(reference));
    a.optimizer.backward(loss(actual.columns),true);auto input_gradient=x.values.grad().clone();x.values.mutable_grad()=Tensor{};
    b.optimizer.backward(loss(reference));close(input_gradient,x.values.grad());
    a.optimizer.step();b.optimizer.step();owners(a.registry,b.registry,a.optimizer,b.optimizer);
    a.q=detach(actual.continuation);expected=detach(reference.continuation);
  }
  auto x=input(a.spec,3,a.q);auto invalid=a.q;invalid.cut-=1;
  rejects([&]{a.engine.run_ports(invalid,x.layout,x.ports,x.stop);});
  invalid=a.q;invalid.states[{2,0}]=invalid.states.begin()->second;
  rejects([&]{col::import_state(a.spec,a.model,invalid);});
  invalid=a.q;invalid.states.begin()->second.slots["unexpected"]=at::zeros({});
  rejects([&]{col::import_state(a.spec,a.model,invalid);});
  invalid=a.q;invalid.ledger.begin()->second.second=invalid.cut;
  rejects([&]{col::import_state(a.spec,a.model,invalid);});
  invalid=a.q;invalid.pending.push_back({0,0,invalid.cut,1,0,0,at::zeros({3},at::kDouble)});
  rejects([&]{col::import_state(a.spec,a.model,invalid);});
  rejects([&]{a.engine.run_ports(a.q,x.layout,x.ports,a.q.cut/a.spec.stride());});
}
void retained(const std::string& family){
  TestOwner a(family),b(family);Streaming scalar(b.graph,b.model,Options{});
  auto x=input(a.spec,0,a.q);auto first=a.engine.run_ports(a.q,x.layout,x.ports,x.stop);
  auto reference=scalar.run(b.q,x.records,x.stop*a.spec.stride(),x.stop*a.spec.stride());
  auto y=input(a.spec,1,first.continuation);
  auto second=a.engine.run_ports(first.continuation,y.layout,y.ports,y.stop);
  auto expected=scalar.run(reference.continuation,y.records,y.stop*a.spec.stride(),y.stop*a.spec.stride());
  observations(a.spec,second,expected);
  auto objective=[](Tensor value,const Continuation& q){
    for(const auto& [_,s]:q.states)for(const auto& [__,t]:s.slots)if(t.is_floating_point())value=value+.03*t.square().sum();
    return value;
  };
  std::vector<Tensor> ar,br;for(const auto& p:a.registry.owners())ar.push_back(p.value);
  for(const auto& p:b.registry.owners())br.push_back(p.value);
  ar.insert(ar.end(),{x.values,y.values});br.insert(br.end(),{x.values,y.values});
  auto ga=torch::autograd::grad({objective(loss(second.columns),second.continuation)},ar,{},true,false,true);
  auto gb=torch::autograd::grad({objective(loss(expected),expected.continuation)},br,{},false,false,true);
  for(size_t i=0;i<ga.size();++i)close(ga[i],gb[i]);
  require(ga[ga.size()-2].defined(),"first window input detached by cursor codec");
}
void process_restore(const std::string& executable,const std::filesystem::path& directory,const std::string& family){
  require(std::filesystem::create_directory(directory),"family checkpoint directory exists");
  TestOwner a(family);a.step(0);a.step(1);a.save(directory/"initial.pt");a.step(2);
  const auto child=fork();require(child>=0,"cannot fork standalone checkpoint verifier");
  if(!child){execl(executable.c_str(),executable.c_str(),"--restore",directory.c_str(),family.c_str(),nullptr);_exit(127);}
  int status=0;require(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"new-process column restore failed");
  TestOwner b(family);b.restore(directory/"final.pt");state(a.q,b.q,true);
  // Checkpoint gradients are not optimizer state. Recompute both next windows
  // from exactly restored raw/parameters/slots, then compare their gradients too.
  a.step(3);b.step(3);state(a.q,b.q,true);owners(a.registry,b.registry,a.optimizer,b.optimizer,true);
}
}
int main(int argc,char** argv){
  at::set_num_threads(1);const auto directory=std::filesystem::temp_directory_path()/("tide-column-owner-"+std::to_string(getpid()));
  try{
    if(argc==4&&std::string(argv[1])=="--restore"){
      TestOwner owner(argv[3]);owner.restore(std::filesystem::path(argv[2])/"initial.pt");owner.step(2);
      owner.save(std::filesystem::path(argv[2])/"final.pt");return 0;
    }
    require(std::filesystem::create_directory(directory),"test checkpoint directory exists");
    const auto executable=std::filesystem::absolute(argv[0]).string();
    for(const auto& family:{"kv","affine","delta","gated_delta","ssd","heterogeneous"}){
      check(family,"fc");process_restore(executable,directory/family,family);
    }
    check("kv","hb");
    retained("kv");retained("ssd");
    std::filesystem::remove_all(directory);
    std::cout<<"column owner: independent streaming/ports/int64, three SGD steps, exact new-process restore passed\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';std::filesystem::remove_all(directory);return 1;}
}
