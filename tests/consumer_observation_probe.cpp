// Exercise the production post-run guard/record without device allocations.
#include "../tools/online_bench/capacity_record.h"
#include <iostream>
int main() {
  using namespace tide_flow::capacity;
  try {
    I count;std::cin>>count;
    if(count<1||count>16)throw std::invalid_argument("probe device count");
    Plan plan{};std::vector<I> peaks;
    for(I i=0;i<count;++i){I estimated,observed;std::cin>>estimated>>observed;
      plan.cards.push_back({i,estimated});peaks.push_back(observed);}
    if(!std::cin)throw std::invalid_argument("probe input");
    std::cout<<'{';observation_fields(std::cout,plan,peaks);std::cout<<'}';
    return within_estimate(plan,peaks)?0:2;
  }catch(const std::exception& e){std::cerr<<e.what();return 3;}
}
