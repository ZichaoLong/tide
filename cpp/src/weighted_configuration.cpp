// Pure value comparison for binding inspection and numerical grouping.
#include "tide/weighted.h"
#include <tuple>
namespace tide::weighted {
bool same_configuration(const KVConfig& a,const KVConfig& b){
  auto key=[](const KVConfig& c){return std::tie(c.width,c.summary_width,c.adapter_rank,c.heads,c.ffn_width,c.window,
    c.memory_read,c.beta_short,c.epsilon,c.alpha_att,c.alpha_ff,c.position_origin,c.position_scale,
    c.full_frequencies,c.read_frequencies,c.distance_slopes);};return key(a)==key(b);
}
bool same_configuration(const RecurrentConfig& a,const RecurrentConfig& b){
  auto key=[](const RecurrentConfig& c){return std::tie(c.memory_family,c.memory_read,c.width,c.summary_width,c.key_width,
    c.value_width,c.ffn_width,c.heads,c.groups,c.conv_size,c.conv_stride,c.lambda_short,c.epsilon,c.alpha_mem,c.alpha_ff,
    c.qk_epsilon,c.output_epsilon,c.matrix_rates);};return key(a)==key(b);
}
bool same_configuration(const RegionalConfig& a,const RegionalConfig& b){
  auto key=[](const RegionalConfig& c){return std::tie(c.hidden_width,c.region_control_width,c.k,c.gamma_max,
    c.allocation_temperature,c.total_fraction,c.ent_alpha,c.eta,c.fixed_gain,c.gain_logit,c.selection_mode,
    c.region_control_mode,c.region_evidence,c.regularizer,c.send_gain,c.allocation_base,c.local_score,c.region_score,
    c.selection_width,c.region_key,c.query_weight,c.alpha_match,c.alpha_res,c.alpha_loc,c.alpha_sel);};return key(a)==key(b);
}
}
