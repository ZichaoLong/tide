#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace tide::device_online {
// Static shape choices only. The actual bucket is selected by device counts.
inline std::vector<int64_t> weighted_bank_capacities(int64_t rows){
  if(rows<1)throw std::invalid_argument("banked GEMM row capacity must be positive");
  std::vector<int64_t> out;
  for(int64_t n=1;;n=n>rows/2?rows:n*2){out.push_back(n);if(n==rows)break;}
  return out;
}
inline void weighted_bank_shape(int64_t rows,int64_t in,int64_t out,int64_t banks){
  if(rows<1||in<1||out<1||banks<1||banks==std::numeric_limits<int64_t>::max()
      ||static_cast<long double>(banks)*in*out>std::numeric_limits<int64_t>::max()/4.L
      ||static_cast<long double>(rows)*(in+static_cast<long double>(out))>std::numeric_limits<int64_t>::max()/4.L)
    throw std::invalid_argument("invalid or overflowing banked GEMM dimensions");
}
inline long double weighted_bank_control_bytes(int64_t rows,int64_t banks){
  // CANN's local-to-global branch table/selected index, unconditional exits,
  // and a conservative allowance for backend control-map tensors.
  (void)banks;return 16.L*weighted_bank_capacities(rows).size()+96;
}
inline long double weighted_projection_forward_bytes(int64_t rows,int64_t in,int64_t out,int64_t banks){
  weighted_bank_shape(rows,in,out,banks);
  if(banks==1)return rows*(4.L*(2.L*in+out+1)+28)+4.L*(2.L*in*out+out+in);
  // Saved bank + transpose, row operands/results, stable device permutation,
  // and one reusable packed group. There is no rows*out*in term.
  return rows*(4.L*(3.L*in+2.L*out+1)+48)+banks*(8.L*in*out+4.L*out+32)
    +4.L*(static_cast<long double>(in)*out+out)+64+weighted_bank_control_bytes(rows,banks);
}
inline long double weighted_projection_reverse_bytes(int64_t rows,int64_t in,int64_t out,int64_t banks,bool bias){
  weighted_bank_shape(rows,in,out,banks);
  if(banks==1)return rows*(4.L*(in+2.L*out)+9)+4.L*(static_cast<long double>(out)*in+(bias?out:0))+5;
  return rows*(4.L*(3.L*in+3.L*out)+32)+banks*(4.L*(static_cast<long double>(out)*in+(bias?out:0))+32)
    +4.L*(2.L*in*out+(bias?out:0))+64+weighted_bank_control_bytes(rows,banks);
}
inline long double weighted_head_forward_bytes(int64_t rows,int64_t in,int64_t hidden,int64_t out,int64_t banks){
  return weighted_projection_forward_bytes(rows,in,hidden,banks)+weighted_projection_forward_bytes(rows,hidden,out,banks)+4.L*rows*hidden;
}
inline long double weighted_head_reverse_bytes(int64_t rows,int64_t in,int64_t hidden,int64_t out,int64_t banks){
  return weighted_projection_reverse_bytes(rows,in,hidden,banks,true)+weighted_projection_reverse_bytes(rows,hidden,out,banks,true)+4.L*rows*(hidden+1.L);
}
}
