// KV §3–4 / allocation §3: unequal banks, independent roots and a direct formula.
#include "weighted_graph_fixture.h"
#include "tide/weighted_sequence_columns.h"
#include "tide/packing_work.h"
#include "tide/autograd_census.h"
#include "../src/weighted_column_blocks.h"
#include "../src/weighted_regional_packing.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/autograd/autograd.h>
#include <iostream>

namespace {
using namespace tide;
namespace col=weighted::columns;
void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
void close(const Tensor& a,const Tensor& b,const std::string& name){
  require(a.defined()==b.defined(),name+": gradient connection");if(!a.defined())return;
  require(a.sizes()==b.sizes()&&a.requires_grad()==b.requires_grad(),name+": metadata/history");
  const bool fp32=a.scalar_type()==at::kFloat;
  require(at::allclose(a,b,fp32?2e-5:2e-9,fp32?2e-6:2e-11),name+": numerical result");
}
Tensor leaf(std::vector<Index> shape,double offset,at::ScalarType dtype){
  Index n=1;for(auto x:shape)n*=x;
  return (.11*at::sin(at::arange(n,at::kDouble)*.37+offset)+.2).reshape(shape).to(dtype).set_requires_grad(true);
}
void add_heads(NodeWeights& w,const weighted::RegionalConfig& r,at::ScalarType dtype,Index node){
  auto add=[&](const std::string& n,std::vector<Index> s){w.extra[n]=leaf(s,.1*w.extra.size()+node*.31,dtype);};
  auto encoder=[&](const std::string& p,Index in){add(p+".in_weight",{3,in});add(p+".hidden_bias",{3});
    add(p+".out_weight",{2,3});add(p+".out_bias",{2});};
  auto group=[&](const std::string& p){encoder(p+".encoders.cur",2);encoder(p+".encoders.mem",4);encoder(p+".encoders.stat",3);};
  auto head=[&](const std::string& p){add(p+".matrix",{3,13});add(p+".hidden_bias",{3});add(p+".vector",{3});add(p+".bias",{});};
  if(r.local_score=="match"){group("read_program.score");for(const auto* s:{"gate","up"}){
      add(std::string("read_program.score.")+s,{3,6});add(std::string("read_program.score.")+s+"_bias",{3});}}
  if(r.region_evidence=="grouped"){group("read_program.evidence");for(const auto* s:{"cur","mem","stat","interaction"})add(std::string("read_program.evidence.")+s,{2,2});add("read_program.evidence.bias",{2});}
  if(r.region_score=="att"){
    add("read_program.current_weight",{2,2});add("read_program.current_bias",{2});
    if(r.region_key=="content")encoder("read_program.key",4);
    else {add("read_program.key_weight",{2,6});add("read_program.key_bias",{2});}
  }
  if(r.send_gain=="context")head("read_program.gain");else if(r.gain_logit)add("read_program.gain_logit",{});
  for(const auto* s:{"read_q","read_k","read_v"})add(std::string("kernel.")+s,{2,2});
}
weighted::RegionalConfig config(Index variant){
  weighted::RegionalConfig r(variant>=4?"direct_attention":"regional_total");r.hidden_width=3;r.gamma_max=2;
  if(variant==1||variant==2){r.local_score="match";r.selection_width=2;r.alpha_match=.7;r.alpha_res=.3;r.region_evidence="grouped";}
  if(variant==2||variant==3||variant==6){r.region_score="att";r.selection_width=2;r.region_key=variant==3?"content":"content_stats";
    r.query_weight="received";r.alpha_loc=.8;r.alpha_sel=.4;}
  if(variant==3){r.region_control_mode="fixed";r.region_evidence.reset();r.total_fraction=.4;}
  if(variant==4){r.send_gain="context";r.fixed_gain.reset();}
  if(variant==5){r.gain_logit=.2;r.fixed_gain.reset();}
  r.validate();return r;
}
struct Fixture {
  std::vector<NodeWeights> owners;
  std::vector<col::KVReadRequest> reads;
  std::vector<col::KVFullRequest> fulls;
  std::vector<Tensor> roots;
};
Fixture fixture(const std::string& memory,Index variant,at::ScalarType dtype,bool frozen=false,bool positional=false){
  Fixture f;auto model=weighted_check::model(weighted_check::graph("fc","kv"),"kv",at::kCPU,dtype);
  auto r=config(variant);f.owners.assign(model.nodes.begin(),model.nodes.begin()+4);
  for(Index i=0;i<4;++i)add_heads(f.owners[i],r,dtype,i);
  for(Index i=1;i<4;++i)for(const auto* name:{"kernel.summary","full_program.content.out"})f.owners[i].extra[name]=f.owners[0].extra.at(name);
  f.owners[1].extra["kernel.read_k"]=f.owners[1].extra.at("kernel.read_q");
  f.owners[1].extra["kernel.read_v"]=f.owners[0].extra.at("kernel.read_v");
  f.owners[1].extra["full_program.content.query"]=f.owners[0].extra.at("full_program.content.query");
  const Index t=Index{1}<<54;
  const std::vector<col::Layout> query{{{0,3,5},{t+10,t+12,t+20,t+11,t+25}},{{0,1,3},{t+15,t+11,t+13}},
    {{0,2,3},{t+11,t+19,t+14}},{{0,3,5},{t+10,t+12,t+20,t+11,t+25}}};
  const std::vector<col::Layout> banks{{{0,3,5},{t+8,t+12,t+19,t+9,t+23}},{{0,1,3},{t+8,t+9,t+11}},
    {{0,1,2},{t+100,t+100}},{{0,3,5},{t+8,t+12,t+19,t+9,t+23}}};
  for(Index i=0;i<4;++i){auto& w=f.owners[i];auto c=std::get<weighted::KVConfig>(weighted::node_binding(w).memory);c.memory_read=memory;c.window=9;
    if(positional){c.read_frequencies={.13};c.full_frequencies={{.17}};c.distance_slopes={.03};c.position_scale=4.;}
    col::KVWindow window;window.layout=query[i];window.bank_layout=banks[i];
    window.content=leaf({static_cast<Index>(query[i].times.size()),2},.1+i,dtype);
    window.mass=leaf({static_cast<Index>(query[i].times.size())},.7+i,dtype);
    col::KVState initial;initial.layout=i==2?col::Layout{{0,0,0},{}}:col::Layout{{0,1,2},{t+7,t+10}};
    for(const auto* name:{"weights","keys","values","summaries"}){
      const auto dims=[&](Index count){return std::string(name)=="weights"?std::vector<Index>{count}:std::vector<Index>{count,2};};
      window.bank[name]=leaf(dims(banks[i].times.size()),.8+i,dtype);initial.fields[name]=leaf(dims(initial.layout.times.size()),.9+i,dtype);
    }
    if(frozen){for(auto& [name,p]:w.extra)p=p.detach();window.content=window.content.detach();
      for(auto* bank:{&window.bank,&initial.fields})for(auto& [name,p]:*bank)if(name!="weights")p=p.detach();}
    f.reads.push_back({&w,c,r,window,"proposal",initial});
    auto controls=at::ones_like(window.mass)*.4;controls[0]=0;
    if(i==1)controls[controls.numel()-1]=0;
    if(i==3)controls=at::zeros_like(controls);controls.set_requires_grad(true);
    f.fulls.push_back({&w,c,window,controls});
  }
  // Parameter sharing and repeated physical uses are separate from node state.
  f.reads[3].window.content=f.reads[0].window.content;
  auto root=[&](const Tensor& t){if(!t.requires_grad())return;for(const auto& x:f.roots)if(x.is_same(t))return;f.roots.push_back(t);};
  for(const auto& w:f.owners)for(const auto& [name,p]:w.extra)root(p);
  for(const auto& q:f.reads){root(q.window.content);root(q.window.mass);for(const auto* bank:{&q.window.bank,&q.initial.fields})for(const auto& [name,p]:*bank)root(p);}
  for(const auto& q:f.fulls){root(q.window.content);root(q.controls);}
  return f;
}
void vjp(const Tensor& a,const Tensor& b,const std::vector<Tensor>& roots,const std::string& tag){
  require(a.requires_grad()==b.requires_grad(),tag+": field connection");if(!a.requires_grad())return;
  for(double factor:{1.,0.}){
    auto cotangent=(at::arange(a.numel(),a.options()).reshape(a.sizes())*.017+.2)*factor;
    auto ga=torch::autograd::grad({a},roots,{cotangent},true,false,true),gb=torch::autograd::grad({b},roots,{cotangent},true,false,true);
    for(size_t i=0;i<roots.size();++i)close(ga[i],gb[i],tag+" root "+std::to_string(i));
  }
}
Tensor independent_full(const col::KVFullRequest& r,Index row){
  const auto& w=*r.weights;const auto& c=r.config;const auto& window=r.window;
  Index sample=0;while(window.layout.offsets[sample+1]<=row)++sample;
  std::vector<Index> visible;for(Index j=window.bank_layout.offsets[sample];j<window.bank_layout.offsets[sample+1];++j){
    const auto age=window.layout.times[row]-window.bank_layout.times[j];if(age>=0&&(!c.window||age<c.window))visible.push_back(j);}
  require(!visible.empty(),"independent formula needs active history");auto ids=at::tensor(visible,at::kLong);
  auto norm=[&](const Tensor& x){return x/at::sqrt(c.epsilon+x.square().mean(-1,true));};
  auto p=[&](const std::string& name){return w.extra.at("full_program.content."+name);};
  auto u=window.content[row],q=at::matmul(p("query"),norm(u));
  auto keys=window.bank.at("keys").index_select(0,ids),values=window.bank.at("values").index_select(0,ids);
  auto weights=window.bank.at("weights").index_select(0,ids);
  std::vector<Index> ages;for(auto j:visible)ages.push_back(window.layout.times[row]-window.bank_layout.times[j]);
  auto displacement=-at::tensor(ages,at::kLong).to(u.scalar_type())/c.position_scale;
  std::vector<Tensor> memories;const Index dim=c.width/c.heads;
  for(Index h=0;h<c.heads;++h){auto kh=keys.slice(1,h*dim,(h+1)*dim),vh=values.slice(1,h*dim,(h+1)*dim),qh=q.slice(0,h*dim,(h+1)*dim);
    if(!c.full_frequencies.empty()&&!c.full_frequencies[h].empty()){
      require(dim==2&&c.full_frequencies[h].size()==1,"literal test supports one rotary pair per head");
      auto phase=displacement*c.full_frequencies[h][0],a=kh.select(1,0),b=kh.select(1,1);
      kh=at::stack({a*phase.cos()-b*phase.sin(),a*phase.sin()+b*phase.cos()},1);
    }
    auto logits=at::matmul(kh,qh)/std::sqrt(double(dim));if(!c.distance_slopes.empty())logits=logits+c.distance_slopes[h]*displacement;
    auto probability=at::softmax(logits+weights.log(),0);memories.push_back((probability.unsqueeze(-1)*vh).sum(0));
  }
  auto memory=at::cat(memories)*weights.sum().clamp_max(1);
  auto x=u+c.alpha_att*at::matmul(p("out"),memory),n=norm(x);
  return x+c.alpha_ff*at::matmul(p("down"),at::silu(at::matmul(p("gate"),n))*at::matmul(p("up"),n));
}
void check(const std::string& memory,Index variant,at::ScalarType dtype,bool frozen=false,bool positional=false){
  auto f=fixture(memory,variant,dtype,frozen,positional);const Index budget=16*1024*1024;
  for(const auto* mode:{"content","old","proposal"}){
    for(auto& r:f.reads)r.mode=mode;
    work::packing::reset(true);autograd_census::reset(true);auto actual=col::read_many(f.reads,budget);
    auto counts=work::packing::metrics();auto graphs=autograd_census::metrics();work::packing::reset(false);autograd_census::reset(false);
    require(graphs.at("graphs/private/kv.tensor.read/unique_nodes")>0,"private numerical graph missing from census");
    require(counts.at("packing/bucket/kv_tensor_read/rows")==4,"new tensor path not selected");
    bool combined=false;for(const auto& [name,value]:counts)if(name.find("packing/bucket/kv_tensor_read/length/")==0&&
      (name.find("/rows/3/calls")!=std::string::npos||name.find("/rows/4/calls")!=std::string::npos))combined|=value>0;
    require(combined,"unequal Q/K were split into exact-length groups");
    std::vector<TensorFields> expected;for(const auto& r:f.reads)expected.push_back(col::read(*r.weights,r.config,r.regional,r.window,r.mode,r.initial));
    for(size_t i=0;i<actual.size();++i)for(const auto& [name,value]:actual[i]){
      close(value,expected[i].at(name),"Read "+name);if(i<3)vjp(value,expected[i].at(name),f.roots,"Read "+name+" node "+std::to_string(i));}
    vjp(actual[0].at("score").sum()+actual[1].at("score").sum()+actual[3].at("score").sum(),
      expected[0].at("score").sum()+expected[1].at("score").sum()+expected[3].at("score").sum(),f.roots,"Read shared roots");
    {at::NoGradGuard guard;auto inference=col::read_many(f.reads,budget);
      for(size_t i=0;i<actual.size();++i)for(const auto& [name,value]:inference[i])close(value,expected[i].at(name).detach(),"inference "+name);}
  }
  auto actual=col::full_many(f.fulls,budget);std::vector<col::Message> expected;
  for(const auto& r:f.fulls)expected.push_back(col::full(*r.weights,r.config,r.window,r.controls));
  for(size_t i=0;i<actual.size();++i){require(actual[i].rows==expected[i].rows,"Full actual support");
    close(actual[i].content,expected[i].content,"Full");close(actual[i].strength,expected[i].strength,"Full strength");
    if(!actual[i].rows.empty())vjp(actual[i].content,expected[i].content,f.roots,"Full node "+std::to_string(i));}
  auto literal=independent_full(f.fulls[0],actual[0].rows[0]);close(actual[0].content[0],literal,"literal Full");
  vjp(actual[0].content[0],literal,f.roots,"literal Full VJP");
  auto guarded=col::read_many(f.reads,budget);
  {at::NoGradGuard guard;f.reads[0].window.mass.add_(.01);}
  bool rejected=false;try{torch::autograd::grad({guarded[0].at("score").sum()},f.roots,{},false,false,true);}catch(const std::exception&){rejected=true;}
  require(rejected,"original input mutation was not guarded");
}
void check_position_guards(){
  for(bool phase:{false,true}){
    auto f=fixture("qry",0,at::kFloat,false,true);auto r=f.reads[0];
    if(phase)r.config.read_frequencies={10000.};
    else r.window.bank_layout.times[0]-=Index{1}<<25;
    work::packing::reset(true);auto actual=col::read_many({r},16*1024*1024).front();
    auto counts=work::packing::metrics();work::packing::reset(false);
    require(counts.at(std::string("packing/skipped/kv_tensor_position_")+(phase?"phase_span":"integer_span")+"/rows")>0,"position fallback was not explicit");
    auto expected=col::read(*r.weights,r.config,r.regional,r.window,r.mode,r.initial);
    for(const auto& [name,value]:actual){close(value,expected.at(name),"position fallback");vjp(value,expected.at(name),f.roots,"position fallback VJP");}
  }
}
void check_graph_size(){
  auto f=fixture("qry",0,at::kDouble);Index expected=-1;
  for(const auto shape:{std::array<Index,4>{1,1,2,2},std::array<Index,4>{5,3,7,4}}){
    auto r=f.reads[0];const auto [g,b,q,k]=shape;r.window.layout={{0},{}};r.window.bank_layout={{0},{}};
    for(Index sample=0;sample<b;++sample){
      for(Index j=0;j<q;++j)r.window.layout.times.push_back(10+j);
      for(Index j=0;j<k;++j)r.window.bank_layout.times.push_back(8+j);
      r.window.layout.offsets.push_back((sample+1)*q);r.window.bank_layout.offsets.push_back((sample+1)*k);
    }
    r.window.content=leaf({b*q,2},.1,at::kDouble);r.window.mass=leaf({b*q},.2,at::kDouble);
    for(const auto* name:{"weights","keys","values","summaries"})r.window.bank[name]=leaf(
      std::string(name)=="weights"?std::vector<Index>{b*k}:std::vector<Index>{b*k,2},.3,at::kDouble);
    autograd_census::reset(true);auto values=col::read_many(std::vector<col::KVReadRequest>(g,r),16*1024*1024);
    auto metrics=autograd_census::metrics();autograd_census::reset(false);
    const auto nodes=metrics.at("graphs/private/kv.tensor.read/unique_nodes");
    if(expected<0)expected=nodes;else require(nodes==expected,"private graph grew with G/B/Q/K");
    std::vector<Tensor> roots{r.window.content,r.window.bank.at("weights"),r.weights->extra.at("kernel.summary")};
    std::vector<Tensor> terms;for(const auto& value:values)terms.push_back(value.at("score").sum());
    auto single=col::read(*r.weights,r.config,r.regional,r.window,r.mode,r.initial);
    vjp(at::stack(terms).sum(),g*single.at("score").sum(),roots,"repeated shared roles");
  }
  std::cout<<"KV tensor private Read graph nodes (G/B/Q/K independent): "<<expected<<'\n';
}
void check_multihead(){
  for(auto dtype:{at::kDouble,at::kFloat}){
    auto f=fixture("qry",0,dtype);std::vector<Tensor> roots;
    for(size_t i=0;i<f.fulls.size();++i){auto& r=f.fulls[i];auto& w=f.owners[i];r.config.width=4;r.config.heads=2;
      r.config.full_frequencies={{.11},{.23}};r.config.distance_slopes={.02,.07};r.config.position_scale=4.;
      r.window.content=leaf({r.window.content.size(0),4},.2+i,dtype);
      for(const auto* name:{"keys","values"})r.window.bank[name]=leaf({r.window.bank.at(name).size(0),4},.3+i,dtype);
      for(const auto* name:{"query","out"})w.extra[std::string("full_program.content.")+name]=leaf({4,4},.1+i,dtype);
      for(const auto* name:{"gate","up"})w.extra[std::string("full_program.content.")+name]=leaf({3,4},.4+i,dtype);
      w.extra["full_program.content.down"]=leaf({4,3},.5+i,dtype);
      roots.insert(roots.end(),{r.window.content,r.window.bank.at("keys"),r.window.bank.at("values"),r.window.bank.at("weights")});
      for(const auto* name:{"query","out","gate","up","down"})roots.push_back(w.extra.at(std::string("full_program.content.")+name));
    }
    auto actual=col::full_many(f.fulls,16*1024*1024);
    for(size_t i=0;i<f.fulls.size();++i){const auto& r=f.fulls[i];auto expected=col::full(*r.weights,r.config,r.window,r.controls);
      close(actual[i].content,expected.content,"multihead Full");if(!actual[i].rows.empty())vjp(actual[i].content,expected.content,roots,"multihead Full VJP");}
    auto literal=independent_full(f.fulls[0],actual[0].rows[0]);close(actual[0].content[0],literal,"multihead literal");
    vjp(actual[0].content[0],literal,roots,"multihead literal VJP");
  }
}
void check_column_owner(){
  auto f=fixture("sum",0,at::kDouble);
  std::vector<col::KVReadRequest> requests(3,f.reads[0]);
  for(size_t i=0;i<requests.size();++i)requests[i].weights=&f.owners[i];
  auto descriptors=col::read_many(requests,16*1024*1024);
  col::RegionalFields sources;std::map<Index,std::vector<Index>> domains;
  std::vector<Tensor> roots=f.roots;
  for(Index i=0;i<3;++i){sources[i]=descriptors[i];domains[i]={0,1,2,3,4};
    for(const auto& [name,value]:descriptors[i]){
      require(value.storage().unsafeGetStorageImpl()==descriptors[0].at(name).storage().unsafeGetStorageImpl(),
              "Read published a separate numerical node field");
      if(value.requires_grad())roots.push_back(value);
    }
  }
  col::RegionalPacking packing(domains);
  work::packing::reset(true);auto actual=packing.groups[0].pack(sources);auto metrics=work::packing::metrics();work::packing::reset(false);
  require(metrics.at("packing/projection/column_block_borrow/calls")==static_cast<Index>(descriptors[0].size()),
          "aligned region did not borrow each actual Read field owner");
  for(const auto& [name,value]:actual.fields){
    require(value.storage().unsafeGetStorageImpl()==descriptors[0].at(name).storage().unsafeGetStorageImpl(),
            "region repacked an aligned owner");
    std::vector<Tensor> values;for(const auto& fields:descriptors)values.push_back(fields.at(name));
    auto expected=at::stack(values,1);close(value,expected,"borrowed field values");
    if(value.requires_grad())vjp((value.select(1,0)-2*value.select(1,1)+value.select(1,2)).square(),
      (expected.select(1,0)-2*expected.select(1,1)+expected.select(1,2)).square(),roots,"cross-node original roots");
  }
  col::RegionalPacking ragged({{0,{0,1,2,3,4}},{1,{0,2,4,6,8}},{2,{1,2,3,5,7}}});
  for(const auto& group:ragged.groups)if(group.candidates==2){
    auto packed=group.pack(sources);std::vector<Index> enabled;
    for(auto node:group.node_ids)enabled.push_back(node!=2);
    auto support=at::tensor(enabled,at::kLong).to(at::kBool).reshape({static_cast<Index>(group.rows.size()),2});
    auto result=group.pullback(sources,{{"score",at::zeros_like(packed.fields.at("score"))}},support);
    require(!result.at(2).at("score").defined(),"missing regional support became connected zero");
    require(result.at(0).at("score").defined()&&result.at(1).at("score").defined(),"used zero disconnected");
    require(result.at(0).at("score").storage().unsafeGetStorageImpl()==result.at(1).at("score").storage().unsafeGetStorageImpl(),
            "regional transpose split its numerical bank by node");
  }
  auto owner=at::arange(12,at::kDouble).reshape({6,2});
  auto a=owner.narrow(0,0,2).detach().set_requires_grad(true),b=owner.narrow(0,2,4).detach().set_requires_grad(true);
  auto borrowed=col::blocks::concatenate({a,b});
  auto independent=owner.narrow(0,0,2).detach().set_requires_grad(true);
  auto bars=torch::autograd::grad({borrowed.sum()},{a,b,independent},{},true,false,true);
  require(bars[0].defined()&&bars[1].defined()&&!bars[2].defined(),"borrow collapsed independent storage aliases");
  for(Index kind=0;kind<3;++kind){
    Tensor x,y;
    if(kind==0){x=owner.as_strided({0,2},{999,7});y=owner;}
    else if(kind==1){x=owner.as_strided({1,2},{999,1});y=owner.narrow(0,1,5);}
    else{x=owner.as_strided({2,0},{999,7});y=owner.as_strided({3,0},{1,17});}
    x=x.detach().set_requires_grad(true);y=y.detach().set_requires_grad(true);
    require(x.is_contiguous()&&y.is_contiguous(),"degenerate stride fixture");
    auto joined=col::blocks::concatenate({x,y}),literal=at::cat({x,y});
    close(joined,literal,"degenerate contiguous owner");vjp(joined,literal,{x,y},"degenerate owner VJP");
    {at::NoGradGuard guard;require(!col::blocks::concatenate({x,y}).requires_grad(),"no-grad borrow retained graph");}
  }
  {at::NoGradGuard guard;b.add_(1);}
  bool rejected=false;try{torch::autograd::grad({borrowed.sum()},{a,b},{},false,false,true);}catch(const std::exception&){rejected=true;}
  require(rejected,"borrowed owner failed original version guard");
}
}
int main(){try{
  at::set_num_threads(1);
  for(auto dtype:{at::kDouble,at::kFloat})for(const auto* memory:{"sum","qry"})for(Index variant=0;variant<7;++variant)check(memory,variant,dtype);
  check("qry",6,at::kDouble,true);
  for(auto dtype:{at::kDouble,at::kFloat})for(Index variant:{0,1})check("qry",variant,dtype,false,true);
  check_position_guards();check_graph_size();check_multihead();check_column_owner();
  std::cout<<"KV tensor boundary: unequal Q/K, Describe candidates, independent roots, aliases, versions and direct Full passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
