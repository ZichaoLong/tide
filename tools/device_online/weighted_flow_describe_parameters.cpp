#include "weighted_flow_parameters.h"
namespace tide::device_online {
WeightedDescribeParameters bind_weighted_describe(WeightedBanks& b,int64_t d,int64_t ds,const weighted::RegionalConfig& c,int64_t available){
  b.limit_bytes(available);
  WeightedDescribeParameters result;
  const auto shapes=weighted_describe_parameter_shapes(1,d,ds,c);
  auto add=[&](const std::string& role,const std::string& parameter){result[role]=b.bind("describe."+role,parameter,shapes.at(role));};
  auto projection=[&](const std::string& role,const std::string& parameter,bool bias=false){
    add(role+".matrix",parameter);if(bias)add(role+".bias",parameter+"_bias");
  };
  auto head=[&](const std::string& role,const std::string& parameter,bool scalar){
    add(role+".matrix",parameter+(scalar?".matrix":".in_weight"));add(role+".hidden_bias",parameter+".hidden_bias");
    add(role+".projection",parameter+(scalar?".vector":".out_weight"));add(role+".bias",parameter+(scalar?".bias":".out_bias"));
  };
  auto encoders=[&](const std::string& prefix){for(const auto& g:{"cur","mem","stat"})head(prefix+".encoders."+g,"read_program."+prefix+".encoders."+g,false);};
  projection("summary","kernel.summary");
  if(c.local_score=="cat")head("score","read_program.score",true);
  else {
    encoders("score");projection("score.gate","read_program.score.gate",true);projection("score.up","read_program.score.up",true);
    projection("score.vector","read_program.score.vector");projection("score.bias","read_program.score.bias");
  }
  if(c.send_gain=="context")head("gain","read_program.gain",true);
  else if(c.gain_logit)projection("gain.logit","read_program.gain_logit");
  if(c.region_control_mode=="context"){
    if(c.region_evidence=="cat"){add("evidence.matrix","read_program.evidence_weight");add("evidence.bias","read_program.evidence_bias");}
    else {encoders("evidence");for(const auto& x:{"cur","mem","stat","interaction","bias"})projection(std::string("evidence.")+x,std::string("read_program.evidence.")+x);}
  }
  if(c.region_score=="att"){
    add("current.matrix","read_program.current_weight");add("current.bias","read_program.current_bias");
    if(c.region_key=="content")head("key","read_program.key",false);
    else {add("key.matrix","read_program.key_weight");add("key.bias","read_program.key_bias");}
  }
  if(result.size()!=shapes.size())throw std::logic_error("Describe parameter role binding incomplete");return result;
}
}
