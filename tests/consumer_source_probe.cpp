#include "../tools/online_bench/source_values.h"
#include <iostream>
#include <iomanip>
#include <string>

int main() {
  std::string name;std::uint64_t seed,index;
  std::cout<<std::setprecision(17);
  while(std::cin>>name>>seed>>index)
    std::cout<<tide_flow::NamedSource(name,seed)(index)<<'\n';
}
