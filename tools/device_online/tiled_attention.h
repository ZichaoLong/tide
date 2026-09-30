#pragma once
#include "cann_program.h"
#include <algorithm>
#include <stdexcept>

namespace tide::device_online {
struct AttentionTiles {int64_t queries,keys,reserved;};
// Fixed reservations include persistent arenas; row_base and row_key cover all
// retained query scratch. Reduce keys before refusing the minimum physical row.
inline AttentionTiles plan_attention_tiles(long double fixed,long double row_base,long double row_key,
    int64_t budget,int64_t rows,int64_t query_limit,int64_t capacity,int64_t key_limit) {
  if(budget<1||rows<1||query_limit<1||capacity<1||key_limit<1)
    throw std::invalid_argument("invalid attention tile limits");
  auto keys=std::min(capacity,key_limit);
  if(keys<capacity)keys=std::min<int64_t>(keys,256);
  if(fixed+row_base+row_key*keys>budget) {
    if(fixed+row_base+row_key>budget)
      throw std::invalid_argument("attention cache and one query/key row exceed workspace budget");
    keys=std::min<int64_t>(keys,static_cast<int64_t>((budget-fixed-row_base)/row_key));
    if(keys<capacity)keys=std::min<int64_t>(keys,256);
  }
  const auto row=row_base+row_key*keys;
  const auto queries=static_cast<int64_t>(std::min<long double>({static_cast<long double>(rows),
    static_cast<long double>(query_limit),(budget-fixed)/row}));
  return {queries,keys,static_cast<int64_t>(fixed+row*queries)};
}
struct TiledAttentionSpec {
  int64_t heads,kv_heads,capacity,owners,keys;
  bool fiber;
  double scale;
  int64_t event_rows=0; // Nonzero selects immutable old + compact node-time KV.
  bool fiber_bias_rows=false; // Each event retains its own repeated-decay bias.
};
// Queries already refer to actual packed work. Dummy query/key padding has
// independent zero storage and never changes a real denominator. work holds
// int64 [executed key tiles, real score entries, padding score entries].
at::Tensor append_tiled_attention(CannProgram&,const at::Tensor& events,const at::Tensor& tokens,
    const at::Tensor& ids,const at::Tensor& query,const at::Tensor& key,const at::Tensor& value,
    const at::Tensor& bias,const at::Tensor& error,const at::Tensor& work,TiledAttentionSpec);
} // namespace tide::device_online
