#include "weighted_describe.h"
namespace tide::device_online {
std::map<std::string,std::vector<int64_t>> weighted_describe_parameter_shapes(
    int64_t banks,int64_t d,int64_t ds,const weighted::RegionalConfig& c){
  c.validate();if(banks<1||d<1||ds<1)throw std::invalid_argument("invalid Describe dimensions");
  std::map<std::string,std::vector<int64_t>> out;const auto hidden=c.hidden_width,feature=d+4*ds+3,select=c.selection_width.value_or(0);
  auto projection=[&](std::string name,int64_t input,int64_t output,bool bias){out[name+".matrix"]={banks,output,input};if(bias)out[name+".bias"]={banks,output};};
  auto head=[&](std::string name,int64_t input,int64_t output){out[name+".matrix"]={banks,hidden,input};out[name+".hidden_bias"]={banks,hidden};
    out[name+".projection"]={banks,output,hidden};out[name+".bias"]={banks,output};};
  auto encoders=[&](std::string name){head(name+".cur",d,select);head(name+".mem",2*ds,select);head(name+".stat",3,select);};
  projection("summary",d,ds,false);
  if(c.local_score=="cat")head("score",feature,1);
  else {encoders("score.encoders");projection("score.gate",3*select,hidden,true);projection("score.up",3*select,hidden,true);
    projection("score.vector",hidden,1,false);projection("score.bias",1,1,false);}
  if(c.send_gain=="context")head("gain",feature,1);else if(c.gain_logit)projection("gain.logit",1,1,false);
  if(c.region_control_mode=="context"){
    const auto dr=*c.region_control_width;
    if(c.region_evidence=="cat")projection("evidence",feature,dr,true);
    else {encoders("evidence.encoders");for(const auto& role:{"cur","mem","stat","interaction"})projection("evidence."+std::string(role),select,dr,false);projection("evidence.bias",1,dr,false);}
  }
  if(c.region_score=="att"){
    projection("current",d,select,true);
    if(c.region_key=="content")head("key",2*ds,select);else projection("key",2*ds+2,select,true);
  }
  return out;
}
}
