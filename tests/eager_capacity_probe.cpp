// Static qualification adapter: never resolves a device or allocates a model.
#include "eager_capacity.h"
#include <iostream>
int main(int argc,char** argv) {
  try {
    const auto c=tide_flow::parse(argc,argv);const auto p=tide_flow::read_packet(c.packet);
    std::vector<tide::Index> budgets(c.devices);
    for(auto& budget:budgets)if(!(std::cin>>budget))throw std::invalid_argument("missing device budget");
    const auto result=tide_flow::eager_capacity::plan(p,c,budgets);
    tide_flow::eager_capacity::record(std::cout,p,c,result);std::cout<<'\n';return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
