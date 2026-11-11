// KV §1.2: raw ragged banks and terminal retention, independently of queries.
#include "weighted_column_layout.h"
#include <numeric>
namespace tide::weighted::columns {
using detail::require;using detail::weight;
void Layout::validate()const {
  require(offsets.size()>=2&&offsets.front()==0&&offsets.back()==static_cast<Index>(times.size()),"invalid column offsets");
  for(Index b=0;b<samples();++b)
    require(offsets[b]>=0&&offsets[b]<=offsets[b+1]&&offsets[b+1]<=static_cast<Index>(times.size()),"invalid column offset bounds");
  for(Index b=0;b<samples();++b){
    require(offsets[b]<=offsets[b+1],"decreasing column offsets");
    for(Index j=offsets[b];j<offsets[b+1];++j)
      require(times[j]>=0&&(j==offsets[b]||times[j]>times[j-1]),"column times must strictly increase");
  }
}
Layout Layout::select(const std::vector<Index>& rows)const {
  Layout out;out.offsets={0};Index b=0,previous=-1;
  for(auto row:rows){
    require(row>previous&&row>=0&&row<static_cast<Index>(times.size()),"column rows must be sorted, unique and in range");
    while(b<samples()&&row>=offsets[b+1]){out.offsets.push_back(out.times.size());++b;}
    out.times.push_back(times[row]);previous=row;
  }
  while(b++<samples())out.offsets.push_back(out.times.size());
  return out;
}
KVState KVState::from_states(const std::vector<State>& states){
  require(!states.empty(),"KV columns require samples");KVState out;out.layout.offsets={0};
  std::map<std::string,std::vector<Tensor>> rows;std::vector<Tensor> inert;
  for(const auto& s:states){
    auto stamps=host_ids(s.slots.at("times"));out.layout.times.insert(out.layout.times.end(),stamps.begin(),stamps.end());
    out.layout.offsets.push_back(out.layout.times.size());out.last_times.push_back(s.last_time);out.observations.push_back(s.observations);
    inert.push_back(s.value);
    for(const auto& name:{"keys","values","summaries","weights"})rows[name].push_back(s.slots.at(name));
  }
  out.inert=at::stack(inert);for(const auto& [name,values]:rows)out.fields[name]=at::cat(values);
  out.layout.validate();return out;
}
std::vector<State> KVState::states()const {
  std::vector<State> out;auto times=indices(layout.times,inert.device());
  for(Index b=0;b<layout.samples();++b){
    State s{inert[b],last_times[b],observations[b],{}};
    for(const auto& [name,t]:fields)s.slots[name]=t.slice(0,layout.offsets[b],layout.offsets[b+1]);
    s.slots["times"]=times.slice(0,layout.offsets[b],layout.offsets[b+1]);out.push_back(s);
  }
  return out;
}
void KVState::validate(const NodeWeights& w,const KVConfig& c)const {
  layout.validate();const Index n=layout.times.size(),b=layout.samples();
  require(last_times.size()==size_t(b)&&observations.size()==size_t(b),"KV clock/count domain differs");
  require(fields.size()==4&&inert.sizes()==at::IntArrayRef({b,c.width+1})&&inert.count_nonzero().item<Index>()==0,"invalid KV inert state");
  std::map<std::string,std::vector<Index>> shapes{{"keys",{n,c.width}},{"values",{n,c.width}},
      {"summaries",{n,c.summary_width}},{"weights",{n}}};
  for(const auto& [name,shape]:shapes){const auto& t=fields.at(name);
    require(t.sizes()==at::IntArrayRef(shape)&&t.device()==w.bias.device()&&t.scalar_type()==w.bias.scalar_type()
      &&at::isfinite(t).all().item<bool>(),"invalid KV raw column shape/metadata/value");}
  require(inert.device()==w.bias.device()&&inert.scalar_type()==w.bias.scalar_type(),"KV inert metadata differs");
  auto weights=fields.at("weights");require(!((weights<0)|(weights>1)).any().item<bool>(),"KV write weight outside [0,1]");
  for(Index i=0;i<b;++i){const auto a=layout.offsets[i],z=layout.offsets[i+1];
    require(last_times[i]>=-1&&observations[i]>=z-a&&(observations[i]==0)==(last_times[i]==-1)
      &&(!c.window||z-a<=c.window)&&(z==a||layout.times[z-1]<=last_times[i]),"invalid KV column clocks/counts");}
}
bool supported(const KVConfig& c){
  c.validate();return true;
}
KVWindow prepare(const NodeWeights& w,const KVConfig& c,const Layout& layout,const Tensor& mass,const Tensor& u,const KVState& left){
  layout.validate();left.validate(w,c);const Index n=layout.times.size();
  require(layout.samples()==left.layout.samples()&&u.sizes()==at::IntArrayRef({n,c.width})&&mass.sizes()==at::IntArrayRef({n}),"KV window dimensions differ");
  require(u.device()==w.bias.device()&&u.scalar_type()==w.bias.scalar_type()&&mass.device()==u.device()&&mass.scalar_type()==u.scalar_type()
    &&at::isfinite(u).all().item<bool>()&&at::isfinite(mass).all().item<bool>()&&!(mass<0).any().item<bool>(),"invalid KV window payload");
  for(Index b=0;b<layout.samples();++b)if(layout.offsets[b]<layout.offsets[b+1])
    require(layout.times[layout.offsets[b]]>left.last_times[b],"KV column window must advance logical time");
  KVWindow out{layout,left.layout,mass,u,left.fields,left};
  auto ids=(mass.detach()>0).nonzero().flatten();auto writes=host_ids(ids);
  if(writes.empty())return out;
  auto x=normalize(u.index_select(0,ids),c.epsilon);
  auto kv=project(x,weight(w,"kernel.base"))+project(project(x,weight(w,"kernel.adapter_v")),weight(w,"kernel.adapter_u"));
  auto split=kv.chunk(2,-1);
  TensorFields fresh{{"keys",split[0]},{"values",split[1]},{"summaries",project(x,weight(w,"kernel.summary"))},{"weights",mass.index_select(0,ids).clamp_max(1)}};
  Layout bank;bank.offsets={0};std::vector<Index> perm;size_t next=0;
  for(Index b=0;b<layout.samples();++b){
    for(Index j=left.layout.offsets[b];j<left.layout.offsets[b+1];++j){perm.push_back(j);bank.times.push_back(left.layout.times[j]);}
    Index count=0;
    while(next<writes.size()&&writes[next]<layout.offsets[b+1]){
      perm.push_back(left.layout.times.size()+next);bank.times.push_back(layout.times[writes[next]]);++next;++count;}
    bank.offsets.push_back(bank.times.size());
    if(count){require(out.terminal.observations[b]<=std::numeric_limits<Index>::max()-count,"KV observation overflow");
      out.terminal.last_times[b]=bank.times.back();out.terminal.observations[b]+=count;}
  }
  auto order=indices(perm,u.device());out.bank_layout=bank;out.terminal.layout=bank;
  for(const auto& [name,t]:fresh)out.bank[name]=at::cat({left.fields.at(name),t}).index_select(0,order);
  out.terminal.fields=out.bank;
  if(c.window){Layout retained;retained.offsets={0};std::vector<Index> keep;
    for(Index b=0;b<layout.samples();++b){
      for(Index j=bank.offsets[b];j<bank.offsets[b+1];++j)
        if(out.terminal.observations[b]==left.observations[b]||out.terminal.last_times[b]-bank.times[j]<c.window){keep.push_back(j);retained.times.push_back(bank.times[j]);}
      retained.offsets.push_back(retained.times.size());}
    auto ids=indices(keep,u.device());out.terminal.layout=retained;
    for(const auto& [name,t]:out.bank)out.terminal.fields[name]=t.index_select(0,ids);
  }
  return out;
}
}
