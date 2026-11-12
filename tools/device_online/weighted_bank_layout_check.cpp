#include "weighted_bank_layout.h"
#include <iostream>
namespace {
void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
template<class F>void rejects(F f){bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}require(caught,"invalid bank shape accepted");}
}
int main(){try{
  using namespace tide::device_online;
  for(int64_t rows:{1,2,3,6,17,128,1025}){
    const auto sizes=weighted_bank_capacities(rows);require(sizes.back()==rows,"missing exact capacity ceiling");
    for(int64_t count=1;count<=rows;++count){int64_t chosen=0;for(auto n:sizes)if(n>=count){chosen=n;break;}
      require(chosen>=count&&chosen<2*count,"bucket doubles actual numerical rows");}
  }
  rejects([]{weighted_bank_capacities(0);});rejects([]{weighted_bank_shape(1,0,2,3);});
  rejects([]{weighted_bank_shape(1,2,3,0);});rejects([]{weighted_bank_shape(INT64_MAX,2,3,4);});
  rejects([]{weighted_bank_shape(1,INT64_MAX,2,4);});rejects([]{weighted_bank_shape(1,2,3,INT64_MAX);});
  constexpr int64_t rows=64,width=32,banks=3,limit=131072;
  const auto f=weighted_projection_forward_bytes(rows,width,width,banks),b=weighted_projection_reverse_bytes(rows,width,width,banks,true);
  require(f<limit&&b<limit&&2.L*rows*width*width*4>limit,"multi-bank capacity distinction lost");
  // Parameter storage grows with banks; doubling rows adds only row buffers.
  const auto d=weighted_projection_forward_bytes(rows*2,width,width,banks)-f;
  require(d<rows*width*width*4.L,"rows still replicate parameter matrices");
  std::cout<<"weighted-bank-layout: passed forward="<<f<<" reverse="<<b<<" rows_doubled_increment="<<d<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
