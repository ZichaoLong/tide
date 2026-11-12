// Compatible groups may have unequal Q/K. Admission charges the padded group.
#include "weighted_kv_tensor.h"
#include "tide/packing_work.h"
#include <numeric>

namespace tide::weighted::columns::tensor {
namespace {
Index width(const Layout& layout){
  Index n=1;for(Index b=0;b<layout.samples();++b)n=std::max(n,layout.offsets[b+1]-layout.offsets[b]);return n;
}
bool same(const Request& a,const Request& b){
  return a.full==b.full&&a.before==b.before&&a.active==b.active&&
    a.query_layout.samples()==b.query_layout.samples()&&same_configuration(a.config,b.config)&&
    (a.full||same_configuration(a.regional,b.regional))&&
    a.window->content.device()==b.window->content.device()&&a.window->content.scalar_type()==b.window->content.scalar_type();
}
struct Estimate {
  Index q=1,k=1,nodes=0;long double original_queries=0;
  std::vector<std::string> names;
  std::vector<Tensor> first;
  std::vector<bool> shared;
  void add(const Request& r){
    q=std::max(q,width(r.query_layout));k=std::max(k,width(*r.bank_layout));
    original_queries+=r.window->layout.times.size();
    if(!nodes){names=parameters(r);for(const auto& name:names)first.push_back(detail::weight(*r.owner,name));shared.assign(names.size(),true);}
    else for(size_t j=0;j<names.size();++j)shared[j]=shared[j]&&first[j].is_same(detail::weight(*r.owner,names[j]));
    ++nodes;
  }
  long double bytes(const Request& r)const{
    long double parameters_size=0;for(size_t j=0;j<first.size();++j)parameters_size+=first[j].numel()*(shared[j]?1:nodes);
    const auto& c=r.config;const long double rows=static_cast<long double>(nodes)*r.query_layout.samples();
    const auto scalar=std::max<Index>(4,r.window->content.element_size());
    return scalar*(4*parameters_size+rows*q*(32*c.width+8*c.ffn_width+16*c.summary_width+128)
      +rows*k*(20*c.width+8*c.summary_width)+rows*q*k*(32+8*c.heads)+2*original_queries*c.width)
      +32*rows*(q+k+static_cast<long double>(q)*k)+8*original_queries;
  }
};
void classify(std::vector<Request>& requests){
  std::map<std::string,std::pair<std::vector<Tensor>,std::vector<size_t>>> devices;
  for(size_t i=0;i<requests.size();++i){auto& r=requests[i];if(!r.full&&r.config.memory_read=="sum")continue;
    std::vector<Index> visible(r.bank_layout->times.size(),0);
    for(Index b=0;b<r.query_layout.samples();++b){
      auto first=r.query_layout.times.begin()+r.query_layout.offsets[b],last=r.query_layout.times.begin()+r.query_layout.offsets[b+1];
      for(Index j=r.bank_layout->offsets[b];j<r.bank_layout->offsets[b+1];++j){const auto time=r.bank_layout->times[j];
        auto q=r.before?std::upper_bound(first,last,time):std::lower_bound(first,last,time);
        if(q!=last&&(!r.config.window||static_cast<long double>(*q)-time<r.config.window))visible[j]=1;
      }
    }
    auto mask=indices(visible,r.window->content.device()).to(at::kBool);
    auto& group=devices[r.window->content.device().str()];
    group.first.push_back(((r.bank->at("weights").detach()>0)&mask).any());group.second.push_back(i);
  }
  for(const auto& [device,group]:devices){
    auto host=at::stack(group.first).to(at::kCPU).contiguous();auto values=host.const_data_ptr<bool>();
    for(size_t i=0;i<group.second.size();++i)requests[group.second[i]].active=values[i];
  }
}
template<class Single,class Publish>
void execute(std::vector<Request> requests,Index budget,Single single,Publish publish){
  detail::require(budget>0,"KV tensor workspace budget must be positive");if(requests.empty())return;
  for(const auto& r:requests){r.query_layout.validate();r.bank_layout->validate();r.window->layout.validate();
    detail::require(r.query_layout.samples()==r.bank_layout->samples(),"KV tensor bank sample domain differs");
    const auto n=static_cast<Index>(r.window->layout.times.size()),k=static_cast<Index>(r.bank_layout->times.size());
    detail::require(r.window->content.sizes()==at::IntArrayRef({n,r.config.width})&&r.window->mass.sizes()==at::IntArrayRef({n}),"KV tensor query shape differs");
    const auto& weights=r.bank->at("weights");
    detail::require(weights.sizes()==at::IntArrayRef({k})&&weights.device()==r.window->content.device()
      &&weights.scalar_type()==r.window->content.scalar_type(),"KV tensor bank weight metadata differs");
  }
  classify(requests);std::vector<Request> accepted;
  for(auto& r:requests){r.coordinates=positions(r);
    if(r.coordinates.reason!="plain"&&r.coordinates.reason!="rebased"){
      work::packing::skipped(("kv_tensor_position_"+r.coordinates.reason).c_str(),r.rows.size());single(r.output);
    }else accepted.push_back(std::move(r));}
  requests=std::move(accepted);std::vector<std::vector<size_t>> compatible;
  for(size_t i=0;i<requests.size();++i){bool found=false;
    for(auto& group:compatible)if(same(requests[group[0]],requests[i])){group.push_back(i);found=true;break;}
    if(!found)compatible.push_back({i});}
  auto run=[&](const std::vector<size_t>& ids,const Estimate& estimate){
    if(ids.size()==1&&estimate.bytes(requests[ids[0]])>budget){single(requests[ids[0]].output);return;}
    std::vector<Request> group;for(auto id:ids)group.push_back(requests[id]);
    auto values=evaluate(group);for(size_t i=0;i<ids.size();++i)publish(requests[ids[i]].output,std::move(values[i]));
  };
  for(const auto& candidates:compatible){std::vector<size_t> group;Estimate estimate;
    for(auto i:candidates){auto trial=estimate;trial.add(requests[i]);
      if(!group.empty()&&trial.bytes(requests[i])>budget){run(group,estimate);group.clear();trial=Estimate{};trial.add(requests[i]);}
      group.push_back(i);estimate=std::move(trial);}
    if(!group.empty())run(group,estimate);
  }
}
}
bool eligible(const KVConfig&,const Tensor& like){
  return like.scalar_type()==at::kFloat||like.scalar_type()==at::kDouble;
}
std::vector<std::string> parameters(const Request& r){
  std::vector<std::string> names;
  auto add=[&](const std::string& p,std::initializer_list<const char*> suffixes){for(auto s:suffixes)names.push_back(p+s);};
  auto head=[&](const std::string& p){add(p,{".matrix",".hidden_bias",".vector",".bias"});};
  auto encoder=[&](const std::string& p){add(p,{".in_weight",".hidden_bias",".out_weight",".out_bias"});};
  auto groups=[&](const std::string& p){for(const auto* g:{"cur","mem","stat"})encoder(p+".encoders."+g);};
  if(r.full){add("full_program.content.",{"out","gate","up","down"});if(r.active)names.push_back("full_program.content.query");return names;}
  names.push_back("kernel.summary");
  if(r.active&&r.config.memory_read=="qry")add("kernel.",{"read_q","read_k","read_v"});
  const auto& c=r.regional;
  if(c.local_score=="cat")head("read_program.score");
  else {groups("read_program.score");add("read_program.score",{".gate",".gate_bias",".up",".up_bias",".vector",".bias"});}
  if(c.regional()&&c.region_control_mode!="fixed"){
    if(c.region_evidence=="cat")add("read_program.",{"evidence_weight","evidence_bias"});
    else {groups("read_program.evidence");add("read_program.evidence",{".cur",".mem",".stat",".interaction",".bias"});}
  }
  if(!c.regional()){
    if(c.send_gain=="context")head("read_program.gain");
    else if(c.gain_logit)names.push_back("read_program.gain_logit");
  }
  if(c.region_score=="att"){
    add("read_program.",{"current_weight","current_bias"});
    if(c.region_key=="content")encoder("read_program.key");
    else add("read_program.",{"key_weight","key_bias"});
  }
  return names;
}
void read_groups(const std::vector<KVReadRequest>& all,const std::vector<size_t>& ids,Index budget,std::vector<TensorFields>& out){
  std::vector<Request> requests;
  for(auto i:ids){const auto& r=all[i];std::vector<Index> rows(r.window.layout.times.size());std::iota(rows.begin(),rows.end(),0);
    requests.push_back({r.weights,r.config,r.regional,&r.window,r.mode=="content"?&r.initial.layout:&r.window.bank_layout,
      r.mode=="content"?&r.initial.fields:&r.window.bank,r.window.layout,std::move(rows),false,r.mode=="old",false,i});}
  execute(std::move(requests),budget,[&](size_t i){const auto& r=all[i];out[i]=columns::read(*r.weights,r.config,r.regional,r.window,r.mode,r.initial);},
    [&](size_t i,TensorFields value){out[i]=std::move(value);});
}
void full_groups(const std::vector<KVFullRequest>& all,const std::vector<size_t>& ids,Index budget,std::vector<Message>& out){
  std::vector<Request> requests;
  for(auto i:ids){const auto& r=all[i];requests.push_back({r.weights,r.config,RegionalConfig{},&r.window,&r.window.bank_layout,&r.window.bank,
    r.window.layout.select(out[i].rows),out[i].rows,true,false,false,i});}
  execute(std::move(requests),budget,[&](size_t i){const auto& r=all[i];out[i]=columns::full(*r.weights,r.config,r.window,r.controls);},
    [&](size_t i,TensorFields value){out[i].content=std::move(value.at("content"));});
}
}
