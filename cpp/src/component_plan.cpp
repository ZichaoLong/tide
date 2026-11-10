#include "component_plan.h"
#include <algorithm>
namespace tide {
ComponentPlan::ComponentPlan(const Graph& g){
  const size_t n=g.regions.size();std::vector<std::set<Index>> out(n),in(n);
  for(const auto& e:g.edges){auto a=g.nodes[e.source].region,b=g.nodes[e.target].region;out[a].insert(b);in[b].insert(a);}
  std::vector<std::vector<Index>> adjacency;for(const auto& edges:out)adjacency.emplace_back(edges.begin(),edges.end());
  std::vector<bool> seen(n);std::vector<Index> order;
  for(size_t root=0;root<n;++root){
    if(seen[root])continue;seen[root]=true;
    std::vector<std::pair<Index,size_t>> stack{{root,0}};
    while(!stack.empty()){
      auto& [node,position]=stack.back();
      if(position==adjacency[node].size()){order.push_back(node);stack.pop_back();continue;}
      auto child=adjacency[node][position++];if(!seen[child]){seen[child]=true;stack.push_back({child,0});}
    }
  }
  std::fill(seen.begin(),seen.end(),false);owner.resize(n);
  for(auto it=order.rbegin();it!=order.rend();++it){
    if(seen[*it])continue;std::vector<Index> group,stack{*it};seen[*it]=true;
    while(!stack.empty()){
      auto node=stack.back();stack.pop_back();group.push_back(node);owner[node]=groups.size();
      for(auto jt=in[node].rbegin();jt!=in[node].rend();++jt)if(!seen[*jt]){seen[*jt]=true;stack.push_back(*jt);}
    }
    std::sort(group.begin(),group.end());bool cycle=group.size()>1||out[group[0]].count(group[0]);
    cyclic.push_back(cycle);
    groups.push_back(std::move(group));
  }
  std::vector<std::vector<Index>> internal(groups.size());
  for(size_t i=0;i<g.edges.size();++i){const auto& e=g.edges[i];auto a=owner[g.nodes[e.source].region],b=owner[g.nodes[e.target].region];
    if(a==b)internal[a].push_back(i);}
  for(size_t i=0;i<groups.size();++i)closures.push_back(cyclic[i]?std::make_shared<ClosurePlan>(g,groups[i],internal[i]):nullptr);
}
}
