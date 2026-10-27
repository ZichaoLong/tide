// Allocation appendix S: applicable fields are part of each function's identity.
#include "weighted_regularizers.h"

namespace tide::weighted {
using namespace detail;
bool RegionalConfig::total() const {return selection_mode=="regional_total"||selection_mode=="regional_total_zero";}
bool RegionalConfig::hard() const {return selection_mode=="hard_selection"||selection_mode=="regional_threshold_hard";}
bool RegionalConfig::regional() const {return total()||selection_mode=="regional_threshold"||selection_mode=="regional_threshold_hard";}
bool RegionalConfig::attention() const {return selection_mode=="direct_attention"||selection_mode=="unit_attention"||selection_mode=="pooled_attention";}
RegionalConfig::RegionalConfig(std::string mode):selection_mode(std::move(mode)){
  if(!regional()){
    region_control_mode.reset();region_control_width.reset();region_evidence.reset();send_gain="fixed";fixed_gain=1.;
  }
  if(attention()){
    k.reset();regularizer.reset();allocation_base=selection_mode=="unit_attention"?"uniform":"mu";
  }
  if(hard()||selection_mode=="regional_total_zero"||selection_mode=="proportional"){
    allocation_temperature.reset();regularizer.reset();
  }
  if(selection_mode=="constrained"||selection_mode=="hard_selection")eta=0.;
}
Index RegionalConfig::capacity(Index size) const {
  require(!k||*k<=size,"sending k exceeds region size");return hard()?*k:size;
}
void RegionalConfig::validate() const {
  require(regional()||attention()||selection_mode=="constrained"||selection_mode=="hard_selection"||selection_mode=="proportional",
    "unknown weighted selection mode");
  require(hidden_width>0&&std::isfinite(gamma_max)&&gamma_max>=1,"invalid weighted region dimensions/gamma_max");
  require((local_score=="cat"||local_score=="match")&&(region_score=="ind"||region_score=="att"),"invalid local_score/region_score");
  const bool allocated=selection_mode=="regional_total"||selection_mode=="regional_threshold"||selection_mode=="constrained";
  if(allocated||attention())require(allocation_temperature&&std::isfinite(*allocation_temperature)&&*allocation_temperature>0,
    "positive sending temperature required");
  else require(!allocation_temperature,"allocation_temperature is inapplicable");
  if(allocated){require(regularizer.has_value(),"regularizer is required");Regularizer rule(*regularizer,ent_alpha);}
  else require(!regularizer&&!ent_alpha,"regularizer/ent_alpha are inapplicable");
  if(attention()){
    require(!k&&!eta,"k/eta are inapplicable to Attention");
    require((selection_mode=="direct_attention"&&(allocation_base=="mu"||allocation_base=="received"))||
      (selection_mode=="unit_attention"&&(allocation_base=="mu"||allocation_base=="uniform"))||
      (selection_mode=="pooled_attention"&&allocation_base=="mu"),"inapplicable allocation_base");
  } else require(k&&*k>0&&!allocation_base,"positive k required; allocation_base is inapplicable");
  if(regional()){
    require(!send_gain&&!fixed_gain&&!gain_logit&&region_control_width&&*region_control_width>0,"invalid regional control fields");
    if(region_control_mode=="context")require((region_evidence=="cat"||region_evidence=="grouped")&&!eta&&!total_fraction,"invalid context control fields");
    else {
      require(region_control_mode=="fixed"&&!region_evidence,"fixed control requires no evidence function");
      if(total())require(total_fraction&&std::isfinite(*total_fraction)&&*total_fraction>=0&&*total_fraction<=1,"invalid total_fraction");
      else require(eta&&std::isfinite(*eta),"fixed threshold requires finite eta");
    }
    require(total()?!eta:!total_fraction,"inapplicable regional eta/total_fraction");
  } else {
    require(!region_control_mode&&!region_control_width&&!region_evidence&&!total_fraction,"regional control is inapplicable");
    if(send_gain=="context")require(!fixed_gain&&!gain_logit,"context gain cannot use fixed gain fields");
    else {
      require(send_gain=="fixed","invalid send_gain");
      if(gain_logit)require(!fixed_gain&&std::isfinite(*gain_logit),"invalid fixed gain_logit");
      else require(fixed_gain&&std::isfinite(*fixed_gain)&&*fixed_gain>=1&&*fixed_gain<=gamma_max,"invalid fixed_gain");
    }
    if(selection_mode=="constrained"||selection_mode=="hard_selection")require(eta&&std::isfinite(*eta),"finite eta required");
    else require(!eta,"eta is inapplicable");
  }
  if(local_score=="match"||region_score=="att"||region_evidence=="grouped")require(selection_width&&*selection_width>0,"positive selection_width required");
  else require(!selection_width,"selection_width is inapplicable");
  auto scale=[](const std::optional<double>& a){return a&&std::isfinite(*a)&&*a>=0;};
  if(local_score=="match")require(scale(alpha_match)&&scale(alpha_res),"invalid match scales");
  else require(!alpha_match&&!alpha_res,"match scales are inapplicable");
  if(region_score=="att"){
    require((region_key=="content"||region_key=="content_stats")&&(query_weight=="mu"||query_weight=="received")
      &&(!regional()||query_weight=="received"),"invalid region_key/query_weight");
    require(scale(alpha_loc)&&scale(alpha_sel),"invalid regional score scales");
  } else require(!region_key&&!query_weight&&!alpha_loc&&!alpha_sel,"Attention score fields are inapplicable");
}
}
