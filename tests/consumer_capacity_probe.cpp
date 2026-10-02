// Torch-free probe: the exact planner consumed by the standalone client.
#include "../tools/online_bench/capacity_record.h"
#include <iostream>
int main() {
  using namespace tide_flow::capacity;
  try {
    Geometry g{};I n,edges;bool aggressive;Capacities c;Chunks chunks;
    std::cin>>g.width>>g.batch>>g.vocab>>g.windows>>g.payload>>g.attention>>g.training>>g.adamw>>g.diagnostics>>g.regions>>g.devices>>g.locality>>n>>edges>>aggressive;
    if(n<0||n>10000||edges<0||edges>1000000)throw std::invalid_argument("probe extent");
    g.sources.resize(n);g.slots.resize(n);g.edges.resize(edges);
    for(auto& x:g.sources)std::cin>>x;for(auto& x:g.slots)std::cin>>x;
    for(auto& [a,b]:g.edges)std::cin>>a>>b;
    std::cin>>c.queue>>c.arrivals>>c.outputs>>c.trace>>c.kv>>c.kv_trace>>c.program;
    for(const auto* key:{"full","emission","aggregate","attention","keys","reverse","head"})std::cin>>chunks[key];
    std::vector<I> budgets(g.devices);for(auto& x:budgets)std::cin>>x;
    if(!std::cin)throw std::invalid_argument("probe input");
    record(std::cout,plan(g,c,chunks,budgets,aggressive));return 0;
  }catch(const std::exception& e){std::cerr<<e.what();return 2;}
}
