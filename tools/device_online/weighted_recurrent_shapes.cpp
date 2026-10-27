#include "weighted_recurrent.h"
namespace tide::device_online {
std::map<std::string,std::vector<int64_t>> weighted_recurrent_parameter_shapes(int64_t banks,const weighted::RecurrentConfig& c){
  c.validate();if(banks<1)throw std::invalid_argument("recurrent bank count must be positive");std::map<std::string,std::vector<int64_t>> s;
  auto projection=[&](std::string name,int64_t out,int64_t in,bool bias=false){s[name+".matrix"]={banks,out,in};if(bias)s[name+".bias"]={banks,out};};
  projection("summary",c.summary_width,c.width);
  if(c.basic()){
    projection("key",c.key_width,c.width);projection("value",c.value_width,c.width);projection("retain",1,c.width,true);
    if(c.memory_family=="delta")projection("write",1,c.width,true);
    projection("full.query",c.key_width,c.width);
    for(int a=0;a<2;++a)projection("full.out."+std::to_string(a),c.width,c.value_width);
    if(c.memory_read=="state"){projection("read.query",c.key_width,c.width);
      for(int a=0;a<2;++a)projection("read.out."+std::to_string(a),c.summary_width,c.value_width);}
  }else{
    const auto hv=c.heads*c.value_width;
    s["conv.weight"]={banks,c.conv_size,c.conv_width()};s["conv.bias"]={banks,c.conv_width()};projection("log_rate",c.heads,1);
    if(c.memory_family=="gated_delta"){
      projection("projection",c.conv_width(),c.width);projection("delta",c.heads,c.width,true);projection("write",c.heads,c.width,true);
      projection("full.output_gate",hv,c.width);
    }else {projection("projection",hv+c.conv_width()+c.heads,c.width);projection("delta_bias",c.heads,1);projection("full.direct",c.heads,1);}
    projection("full.rho",hv,1);projection("full.out",c.width,hv);
    if(c.memory_read=="state")projection("read.out",c.summary_width,hv);
  }
  projection("full.gate",c.ffn_width,c.width);projection("full.up",c.ffn_width,c.width);projection("full.down",c.width,c.ffn_width);return s;
}
}
