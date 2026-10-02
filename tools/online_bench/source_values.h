#pragma once
#include <array>
#include <cstdint>
#include <string_view>

namespace tide_flow {
// The packet's named initializer is three LCG steps modulo 2^31-1. Compose
// those integer affine maps before filling the CPU array; this preserves every
// FP32 bit and avoids the old full-sized int64 intermediate tensors.
class NamedSource {
 public:
  NamedSource(std::string_view name,std::uint64_t seed):key_(seed%modulus) {
    for(unsigned char byte:name)key_=(key_*131+byte)%modulus;
  }
  float operator()(std::uint64_t index) const {
    const auto x=(index%modulus+key_)%modulus;
    // Both factors are below modulus, so the product plus increment fits
    // uint64 (in fact int64). Subtraction is signed before exact FP32 scaling.
    const auto value=(coefficients[0]*x+coefficients[1])%modulus;
    return (static_cast<std::int32_t>(value%65536)-32768)*0x1p-20f;
  }
 private:
  static constexpr std::uint64_t modulus=2147483647;
  inline static constexpr auto coefficients=[] {
    std::array<std::uint64_t,2> result{1,0};
    for(int i=0;i<3;++i) {
      result[0]=(result[0]*1103515245)%modulus;
      result[1]=(result[1]*1103515245+12345)%modulus;
    }
    return result;
  }();
  std::uint64_t key_;
};
} // namespace tide_flow
