#pragma once
// Pinned weighted KV §1 / recurrent §1: raw tuples, not effective projections.
#include "tide/weighted.h"

namespace tide::device_online {
struct WeightedRawField {
  std::string name;
  std::vector<int64_t> shape; // Per-entry tail when ragged; whole field otherwise.
  int64_t offset=0, width=0;
  bool ragged=false;
};
// Numerical groups have one schema; unrelated nodes still have private rows.
// Integer row: last_time, observations, cache_length, mathematical time, times[C].
// Floating row: value, then named fields in fields() order; ragged fields use C slots.
class WeightedRawLayout {
 public:
  WeightedRawLayout(const weighted::KVConfig&,int64_t cache_capacity);
  WeightedRawLayout(const weighted::RecurrentConfig&,int64_t cache_capacity);
  const std::string& family() const{return family_;}
  int64_t capacity() const{return capacity_;}
  int64_t floating_width() const{return floating_width_;}
  int64_t integer_width() const{return 4+capacity_;}
  int64_t horizon() const{return horizon_;} // KV window (0=infinity), conv inclusive lag.
  const std::vector<WeightedRawField>& fields() const{return fields_;}
  const WeightedRawField& field(const std::string&) const;
  bool kv() const{return family_=="kv";}
  bool convolution() const{return family_=="gated_delta"||family_=="ssd";}
 private:
  void add(std::string,std::vector<int64_t>,bool ragged=false);
  std::string family_;
  int64_t capacity_=0,floating_width_=0,horizon_=0;
  std::vector<WeightedRawField> fields_;
};
struct WeightedRawState {at::Tensor floating,integer;};
// Explicit detached CPU import/export. Padding is not a logical state field.
// Decode only slices actual buffers; it never calls a memory/reference function.
WeightedRawState encode_weighted_raw_state(const WeightedRawLayout&,const std::vector<State>&,
                                           int64_t budget=256*1024*1024);
std::vector<State> decode_weighted_raw_state(const WeightedRawLayout&,const WeightedRawState&);
// Explicit boundary transfer; no effective-state conversion or dtype substitution.
WeightedRawState copy_weighted_raw_state(const WeightedRawState&,at::Device);
void validate_weighted_raw_buffers(const WeightedRawLayout&,const WeightedRawState&);
} // namespace tide::device_online
