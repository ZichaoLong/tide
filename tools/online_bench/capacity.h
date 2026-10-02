#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace tide_flow::capacity {
using I=int64_t;
// Wider checked arithmetic keeps int64 extents from wrapping before admission.
using Wide=__int128_t;
constexpr I MiB=1024*1024;
inline I bytes(Wide x) {
  if(x<0||x>std::numeric_limits<I>::max())throw std::invalid_argument("consumer memory extent overflow");
  return I(x);
}
struct Geometry {
  I width,batch,vocab,windows,payload;
  bool attention,training,adamw,diagnostics;
  I regions;
  std::vector<I> sources,slots;
  std::vector<std::pair<I,I>> edges;
  I devices;
  bool locality=true;
  I sample_chunks=1,context_bytes=0;
};
struct Capacities {I queue=1024,arrivals=1024,outputs=1024,trace=4096,kv=128,kv_trace=4096,program=64*MiB;};
using Chunks=std::map<std::string,I>;
struct Card {I index,peak,budget=0,usable=0,headroom=0;std::map<std::string,I> phases,components;};
struct Plan {std::vector<Card> cards;std::vector<I> owners,canonical;Chunks requested,effective;I reductions=0;bool aggressive;};

inline std::vector<I> placement(const Geometry& g) {
  const I n=g.sources.size(),d=g.devices;const Wide w=g.width;
  std::vector<I> cost(n+2,1),order(n+2),owners(n+2),load(d),members(d);
  for(I i=0;i<n;++i)cost[i]=bytes(1+g.payload*(w+g.slots[i]*(w*w+w)));
  std::iota(order.begin(),order.end(),0);
  std::stable_sort(order.begin(),order.end(),[&](I a,I b){return cost[a]>cost[b];});
  for(auto i:order){const I j=std::min_element(load.begin(),load.end())-load.begin();owners[i]=j;load[j]=bytes(Wide(load[j])+cost[i]);++members[j];}
  if(!g.locality)return owners;
  std::vector<std::vector<I>> adjacent(n+2);
  for(auto [a,b]:g.edges)if(a!=b){adjacent[a].push_back(b);adjacent[b].push_back(a);}
  const I limit=*std::max_element(load.begin(),load.end());
  for(int pass=0;pass<2;++pass)for(auto i:order) {
    const I old=owners[i];if(members[old]<=1)continue;I next=old;std::vector<I> affinity(d);
    for(auto x:adjacent[i])++affinity[owners[x]];
    for(I j=0;j<d;++j)if(load[j]<=limit-cost[i]&&affinity[j]>affinity[next])next=j;
    if(next!=old){load[old]-=cost[i];load[next]+=cost[i];--members[old];++members[next];owners[i]=next;}
  }
  for(int pass=0;pass<2;++pass)for(auto i:order) {
    const I old=owners[i];I gain=0,best=-1;std::vector<I> affinity(d);
    for(auto x:adjacent[i])++affinity[owners[x]];
    for(I j=0;j<n+2;++j) {
      const I next=owners[j];if(next==old||load[old]-cost[i]>limit-cost[j]||load[next]-cost[j]>limit-cost[i])continue;
      I back=0,stay=0,shared=0;for(auto x:adjacent[j]){back+=owners[x]==old;stay+=owners[x]==next;shared+=x==i;}
      const I delta=affinity[next]-affinity[old]+back-stay-2*shared;
      if(delta>gain){gain=delta;best=j;}
    }
    if(best>=0){const I next=owners[best];load[old]+=cost[best]-cost[i];load[next]+=cost[i]-cost[best];std::swap(owners[i],owners[best]);}
  }
  return owners;
}
inline std::vector<I> canonical_loads(const Geometry& g) {
  const I n=g.sources.size();const Wide w=g.width;std::vector<I> sizes,load(g.devices);
  for(auto [a,b]:g.edges)if(a<n&&b<n)sizes.push_back(bytes(w*w));
  for(auto s:g.sources){sizes.push_back(g.width);if(g.attention)sizes.insert(sizes.end(),{bytes(3*w*w),bytes(w*w),s});else sizes.insert(sizes.end(),s,1);}
  std::stable_sort(sizes.begin(),sizes.end(),std::greater<I>());
  for(auto size:sizes){auto i=std::min_element(load.begin(),load.end());*i=bytes(Wide(*i)+size);}
  return load;
}
inline std::vector<Card> envelope(const Geometry& g,const Capacities& c,const Chunks& chunk,
    const std::vector<I>& owners,const std::vector<I>& canonical) {
  const Wide w=g.width,b=g.batch,v=g.vocab,p=g.payload,windows=g.windows,n=g.sources.size();
  const Wide trace=g.diagnostics?c.trace:0;std::vector<Card> result;
  for(I device=0;device<g.devices;++device) {
    Wide body=0,slots=0,domain=1;const Wide nodes=std::count(owners.begin(),owners.end(),device);
    for(I i=0;i<n;++i)if(owners[i]==device){++body;slots+=g.slots[i];domain=std::max(domain,Wide(g.sources[i]));}
    const Wide projection=(slots+1)*(w*w+w)*p;
    const Wide state_parameters=g.attention&&body?(body+1)*(4*w*w+4*w)*p+4*(body+1)*domain+body*p:0;
    const Wide parameters=projection+state_parameters+nodes*(2*w*p+128);
    const Wide cache=g.attention&&body?(b*body*c.kv+1)*(2*w+1):0,state=b*nodes*(p*w+17);
    const Wide kv_proposal_saving=g.attention&&body?2*p*(b*body*c.kv+1)*w:0;
    const Wide persistent_state=24*cache+4*state-kv_proposal_saving;const bool coordinator=device==0;
    const Wide routing=coordinator?64*(Wide(c.queue)+c.arrivals+c.outputs+trace)*(5*w+32)
      +64*b*((n+2)*(w+4)+Wide(g.regions)*(g.regions+4))+160*(Wide(g.edges.size())+n+4):0;
    const Wide owner_packets=g.devices>1?128*Wide(c.queue)*(6*w+64):0;
    const Wide journals=g.attention&&g.diagnostics&&body?24*Wide(c.kv_trace)*(2*w+8):0;
    Wide forward_work=96*chunk.at("emission")*w*w+128*(Wide(chunk.at("full"))+chunk.at("aggregate"))*(w+domain);
    if(g.attention&&body)forward_work+=96*chunk.at("attention")*w*w+Wide(chunk.at("attention"))*chunk.at("keys")*(32*w+192)+512*chunk.at("attention")*(w+1);
    const Wide masters=g.training?4*Wide(g.adamw?3:2)*canonical[device]:0;
    const Wide consumer=coordinator?2*p*v*w+(g.training?4*(g.adamw?3:2)*2*v*w:0):0;
    const Wide programs=(1+(g.training?4*windows:2))*c.program+512*MiB;
    Wide snapshot=0,saved_contexts=0,accumulation=0,context_pack=0;
    if(g.sample_chunks>1) {
      snapshot=state+4096;
      if(g.attention&&body)snapshot+=(b*body*c.kv+1)*p*(2*w+1)+8*b*body;
      if(coordinator)snapshot+=9*b*(n+2+g.regions)+Wide(c.queue)*(p*w+49)+16;
      saved_contexts=g.sample_chunks*snapshot;
      if(g.context_bytes) {
        saved_contexts=std::min(saved_contexts,Wide(g.context_bytes));
        context_pack=32*(g.attention?b*body*c.kv:0)+16*MiB+(coordinator?32*Wide(c.queue):0);
      }
      accumulation=g.training?8*Wide(canonical[device])+16*MiB:0;
    }
    const Wide retained_pack=g.training?32*(trace+(g.attention&&body?Wide(c.kv_trace):0))+16*MiB:0;
    const Wide base=parameters+persistent_state+routing+owner_packets+journals+forward_work+masters+consumer+programs+saved_contexts+accumulation+context_pack+retained_pack;
    const Wide construction=base+4*Wide(canonical[device])+parameters;
    const Wide head_fixed=32*MiB+4096+8*Wide(c.outputs)+(g.training?4*Wide(c.outputs)*w+4*(3+(p==2))*v*w:0);
    const Wide head_row=(g.training?32:16)*v+(p+(g.training?12:4))*w+160;
    const Wide head_work=coordinator?head_fixed+chunk.at("head")*head_row:0;
    Wide retained=0,gradients=0,reverse_work=0,communication=0,roots=0,proposal=0;
    if(g.training) {
      retained=projection+windows*(state_parameters+2*nodes*w*p+state+p*cache+journals);
      if(coordinator){retained+=windows*32*(trace+c.queue+c.outputs)*(10*w+64);roots=4*windows*c.outputs*w+(g.sample_chunks>1?12:8)*v*w;}
      gradients=windows*(4*(slots+1)*(w*w+w)+4*(4*body*w*w+body*domain)+32*b*nodes*(w+1)+24*cache);
      if(coordinator)gradients+=windows*64*(trace+c.queue+c.outputs)*(w+32);
      gradients+=4*Wide(canonical[device]);
      reverse_work=windows*chunk.at("reverse")*(256*w*w+128*c.kv*(w+1)+128*domain+4096);
      communication=g.devices>1?(coordinator?g.devices:1)*Wide(128*MiB):0;
      proposal=coordinator?(4*(g.adamw?3:2)+p)*2*v*w+32*v*w:0;
    }
    Card card{device,0};
    card.phases={{"construction",bytes(construction)},{"forward_loss",bytes(base+retained+roots+head_work)}};
    if(g.training){card.phases["backward"]=bytes(base+retained+roots+gradients+reverse_work+communication);
      card.phases["optimizer"]=bytes(base+4*Wide(canonical[device])+roots+proposal+communication);}
    card.components={{"parameters",bytes(parameters)},{"state_and_kv",bytes(persistent_state)},{"routing",bytes(routing)},
      {"owner_packets",bytes(owner_packets)},{"journals",bytes(journals)},{"forward_workspace",bytes(forward_work)},
      {"graph_optimizer",bytes(masters)},{"consumer_parameters_optimizer",bytes(consumer)},
      {"programs_and_vendor_allowance",bytes(programs)},{"retained",bytes(retained)},
      {"roots_and_consumer_gradients",bytes(roots)},{"physical_and_canonical_gradients",bytes(gradients)},
      {"reverse_workspace",bytes(reverse_work)},{"canonical_communication",bytes(communication)},
      {"consumer_proposals",bytes(proposal)},{"head_workspace",bytes(head_work)},
      {"continuation_snapshot_bytes",bytes(snapshot)},{"saved_contexts",bytes(saved_contexts)},
      {"gradient_accumulation",bytes(accumulation)},{"context_pack_workspace",bytes(context_pack)},
      {"retained_pack_workspace",bytes(retained_pack)}};
    for(const auto& [_,value]:card.phases)card.peak=std::max(card.peak,value);result.push_back(std::move(card));
  }
  return result;
}
inline Plan plan(const Geometry& g,const Capacities& c,const Chunks& requested,const std::vector<I>& budgets,bool aggressive) {
  const I n=g.sources.size();
  if(g.width<1||g.batch<1||g.vocab<1||g.windows<1||g.regions<1||g.sample_chunks<1||g.context_bytes<0||(g.payload!=2&&g.payload!=4)
      ||g.devices<1||g.devices>16||g.devices>n+2||g.slots.size()!=size_t(n)||budgets.size()!=size_t(g.devices)
      ||c.queue<1||c.arrivals<1||c.outputs<1||c.trace<0||c.kv<1||c.kv_trace<0||c.program<1)
    throw std::invalid_argument("invalid complete-consumer memory geometry/budget");
  for(auto x:g.sources)if(x<1)throw std::invalid_argument("invalid consumer source count");
  for(auto x:g.slots)if(x<0)throw std::invalid_argument("invalid consumer slot count");
  for(auto x:budgets)if(x<1)throw std::invalid_argument("invalid device memory budget");
  for(const auto* key:{"full","emission","aggregate","attention","keys","reverse","head"})
    if(!requested.count(key)||requested.at(key)<1)throw std::invalid_argument("invalid physical memory chunk");
  for(auto [a,b]:g.edges)if(a<0||b<0||a>=n+2||b>=n+2)throw std::invalid_argument("invalid consumer edge");
  Plan out;out.owners=placement(g);out.canonical=canonical_loads(g);out.requested=out.effective=requested;out.aggressive=aggressive;
  for(;;) {
    out.cards=envelope(g,c,out.effective,out.owners,out.canonical);bool fits=true;std::string why;
    for(size_t i=0;i<out.cards.size();++i) {
      auto& card=out.cards[i];card.budget=budgets[i];card.usable=budgets[i]-budgets[i]/(aggressive?10:4)-128*MiB;
      card.headroom=card.budget-card.usable;
      if(card.peak>card.usable){fits=false;why+="device "+std::to_string(i)+": estimated "+std::to_string(card.peak)+" > usable "+std::to_string(card.usable)+"; ";}
    }
    if(fits)return out;
    bool changed=false;for(auto& [_,value]:out.effective)if(value>1){value=std::max<I>(1,value/2);changed=true;}
    if(!changed)throw std::invalid_argument("complete-consumer memory admission refused at minimum physical rows; "+why);
    ++out.reductions;
  }
}
} // namespace tide_flow::capacity
