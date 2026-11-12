#include "weighted_bank_layout.h"
#include <iostream>
namespace {
void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
template<class F>void rejects(F f){bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}require(caught,"invalid bank shape accepted");}
// Independent inventory of retained tensor shapes, including the CANN branch
// table/selected index. Vendor workspace and allocator rounding are separate.
long double forward_inventory(int64_t r,int64_t i,int64_t o,int64_t b){
  const long double t=r<64?r:64;
  return 4.L*(2.L*r*i+r+2.L*b*o*i+b*o+r*o+t*(i+o)+i*o+o+i)
    +8.L*(3.L*r+2.L*b+3)+4.L*(2.L*r+5)+r+b;
}
long double reverse_inventory(int64_t r,int64_t i,int64_t o,int64_t b,bool bias){
  const long double t=r<64?r:64;
  return 4.L*(r*static_cast<long double>(i+o)+b*static_cast<long double>(o)*i+(bias?b*o:0)
    +2*t*(i+o)+2.L*i*o+(bias?2*o:0))+8.L*(r+2.L*b+3)+4.L*(r+5)+2.L*r+b;
}
}
int main(){try{
  using namespace tide::device_online;
  for(int64_t rows:{1,2,3,6,17,63,64,65,128,1025}){
    const auto tile=weighted_bank_tile(rows);require(tile>0&&tile<=rows&&tile<=64,"invalid fixed tile");
    for(int64_t count=1;count<=rows;++count){const auto padded=((count+tile-1)/tile)*tile;
      require(padded>=count&&padded<count+tile,"tile tail bound differs");}
    for(int64_t banks:{1,3,9})for(int64_t width:{1,7,32}){
      if(banks==1)continue; // Single-bank retains its separate whole-matrix path.
      require(weighted_projection_forward_bytes(rows,width,5,banks)>=forward_inventory(rows,width,5,banks),"forward tile tensors undercounted");
      for(bool bias:{false,true})require(weighted_projection_reverse_bytes(rows,width,5,banks,bias)
        >=reverse_inventory(rows,width,5,banks,bias),"reverse tile accumulation tensors undercounted");
    }
  }
  rejects([]{weighted_bank_tile(0);});rejects([]{weighted_bank_tile(-1);});rejects([]{weighted_bank_shape(1,0,2,3);});
  rejects([]{weighted_bank_shape(1,2,3,0);});rejects([]{weighted_bank_shape(INT64_MAX,2,3,4);});
  rejects([]{weighted_bank_shape(1,INT64_MAX,2,4);});rejects([]{weighted_bank_shape(1,2,3,INT64_MAX);});
  require(weighted_bank_control_bytes(1,2)==weighted_bank_control_bytes(1025,1024),"fixed body grows with rows or banks");
  constexpr int64_t rows=64,width=32,banks=3,limit=131072;
  const auto f=weighted_projection_forward_bytes(rows,width,width,banks),b=weighted_projection_reverse_bytes(rows,width,width,banks,true);
  require(f<limit&&b<limit&&2.L*rows*width*width*4>limit,"multi-bank capacity distinction lost");
  // Parameter storage grows with banks; doubling rows adds only row buffers.
  const auto d=weighted_projection_forward_bytes(rows*2,width,width,banks)-f;
  require(d<rows*width*width*4.L,"rows still replicate parameter matrices");
  require(d==weighted_projection_forward_bytes(rows*3,width,width,banks)-weighted_projection_forward_bytes(rows*2,width,width,banks),
    "tile scratch grew above 64 rows");
  std::cout<<"weighted-bank-layout: passed forward="<<f<<" reverse="<<b<<" rows_doubled_increment="<<d<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
