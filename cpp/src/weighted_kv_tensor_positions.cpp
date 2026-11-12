// KV §2–4: bounded common origins. Only metadata is traversed per node/sample.
#include "weighted_kv_tensor.h"
#include <cmath>
#include <limits>

namespace tide::weighted::columns::tensor {
Coordinates positions(const Request& r){
  Coordinates result;if(!r.active||(!r.full&&r.config.memory_read=="sum"))return result;
  auto frequencies=r.full?r.config.full_frequencies:std::vector<std::vector<double>>{r.config.read_frequencies};
  double frequency=0;for(const auto& head:frequencies)for(auto f:head)frequency=std::max(frequency,std::abs(f));
  // Nonempty zero-frequency vectors still describe an exact identity rotation.
  bool rotated=false;for(const auto& f:frequencies)rotated|=!f.empty();if(!rotated)return result;
  const bool fp64=r.window->content.scalar_type()==at::kDouble;
  const Index integer_limit=Index{1}<<(fp64?53:24);
  const double epsilon=fp64?std::numeric_limits<double>::epsilon():std::numeric_limits<float>::epsilon();
  const double phase_limit=1/(8*std::sqrt(epsilon));
  const auto& q=r.query_layout;const auto& k=*r.bank_layout;
  detail::require(q.samples()==k.samples(),"KV tensor position sample domains differ");
  result.query.resize(q.times.size());result.key.resize(k.times.size());
  for(Index b=0;b<q.samples();++b){
    Index first=std::numeric_limits<Index>::max(),last=std::numeric_limits<Index>::min();
    for(const auto* layout:{&q,&k})if(layout->offsets[b]<layout->offsets[b+1]){
      first=std::min(first,layout->times[layout->offsets[b]]);last=std::max(last,layout->times[layout->offsets[b+1]-1]);}
    if(first==std::numeric_limits<Index>::max())continue;
    const __int128 span=static_cast<__int128>(last)-first;
    if(span>integer_limit){result.reason="integer_span";return result;}
    const double phase=static_cast<double>(span)/r.config.position_scale*frequency;
    if(!std::isfinite(phase)||phase>phase_limit){result.reason="phase_span";return result;}
    for(Index j=q.offsets[b];j<q.offsets[b+1];++j)result.query[j]=q.times[j]-first;
    for(Index j=k.offsets[b];j<k.offsets[b+1];++j)result.key[j]=k.times[j]-first;
  }
  result.reason="rebased";return result;
}
}
