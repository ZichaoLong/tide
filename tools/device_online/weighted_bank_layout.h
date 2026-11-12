#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace tide::device_online {
// Finite baseline, not an optimum claim. One body and bounded scratch for any
// row/bank count; the device advances actual groups and masks partial tails.
inline int64_t weighted_bank_tile(int64_t rows){
  if(rows<1)throw std::invalid_argument("banked GEMM row capacity must be positive");
  return rows<64?rows:64;
}
inline void weighted_bank_shape(int64_t rows,int64_t in,int64_t out,int64_t banks){
  if(rows<1||in<1||out<1||banks<1||banks==std::numeric_limits<int64_t>::max()
      ||static_cast<long double>(banks)*in*out>std::numeric_limits<int64_t>::max()/4.L
      ||static_cast<long double>(rows)*(in+static_cast<long double>(out))>std::numeric_limits<int64_t>::max()/4.L)
    throw std::invalid_argument("invalid or overflowing banked GEMM dimensions");
}
inline long double weighted_bank_control_bytes(int64_t rows,int64_t banks){
  // Two-way branch table/selected index, one unconditional back edge and
  // backend control-map allowance. This does not grow with R or bank count.
  (void)rows;(void)banks;return 128;
}
inline long double weighted_projection_forward_bytes(int64_t rows,int64_t in,int64_t out,int64_t banks){
  weighted_bank_shape(rows,in,out,banks);
  if(banks==1)return rows*(4.L*(2.L*in+out+1)+28)+4.L*(2.L*in*out+out+in);
  // Saved bank + transpose, full row tape/permutation, one tile of operands,
  // bank matrix/bias scratch and possible RMS weight. No rows*out*in term.
  return rows*(4.L*(2.L*in+out+1)+48)+banks*(8.L*in*out+4.L*out+32)
    +4.L*weighted_bank_tile(rows)*(in+static_cast<long double>(out))
    +4.L*(static_cast<long double>(in)*out+out+in)+64+weighted_bank_control_bytes(rows,banks);
}
inline long double weighted_projection_reverse_bytes(int64_t rows,int64_t in,int64_t out,int64_t banks,bool bias){
  weighted_bank_shape(rows,in,out,banks);
  if(banks==1)return rows*(4.L*(in+2.L*out)+9)+4.L*(static_cast<long double>(out)*in+(bias?out:0))+5;
  // Two tile inputs, dX and transposed G; two matrix tiles and (when bound)
  // two bias vectors for deterministic read/add/write of partial gradients.
  return rows*(4.L*(in+static_cast<long double>(out))+32)+banks*(4.L*(static_cast<long double>(out)*in+(bias?out:0))+32)
    +8.L*weighted_bank_tile(rows)*(in+static_cast<long double>(out))
    +4.L*(2.L*in*out+(bias?2.L*out:0))+64+weighted_bank_control_bytes(rows,banks);
}
inline long double weighted_head_forward_bytes(int64_t rows,int64_t in,int64_t hidden,int64_t out,int64_t banks){
  return weighted_projection_forward_bytes(rows,in,hidden,banks)+weighted_projection_forward_bytes(rows,hidden,out,banks)+4.L*rows*hidden;
}
inline long double weighted_head_reverse_bytes(int64_t rows,int64_t in,int64_t hidden,int64_t out,int64_t banks){
  return weighted_projection_reverse_bytes(rows,in,hidden,banks,true)+weighted_projection_reverse_bytes(rows,hidden,out,banks,true)+4.L*rows*(hidden+1.L);
}
}
