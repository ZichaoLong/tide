#pragma once
// Bounded little-endian primitives shared by the independent native formats.
#include "checkpoint_internal.h"
#include <cstring>
#include <limits>
namespace tide::checkpoint_detail {
constexpr uint64_t kMaxString = 1 << 20;
bool little_endian();
uint64_t checksum(const uint8_t*, size_t);
struct Writer {
  std::vector<uint8_t> bytes;

  void room(uint64_t size) const {
    if (size > kMaxFileBytes - bytes.size()) fail("checkpoint is too large");
  }
  void raw(const uint8_t* data, size_t size) {
    room(size);
    if (size) bytes.insert(bytes.end(), data, data + size);
  }
  void u8(uint8_t value) { room(1); bytes.push_back(value); }
  void u32(uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) u8(static_cast<uint8_t>(value >> shift));
  }
  void u64(uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) u8(static_cast<uint8_t>(value >> shift));
  }
  void i64(int64_t value) { u64(static_cast<uint64_t>(value)); }
  void f64(double value) {
    uint64_t bits;
    static_assert(sizeof(bits) == sizeof(value), "unexpected double width");
    std::memcpy(&bits, &value, sizeof(bits));
    u64(bits);
  }
  void string(const std::string& value) {
    if (value.size() > kMaxString) fail("string is too large");
    u64(value.size());
    raw(reinterpret_cast<const uint8_t*>(value.data()), value.size());
  }
  void tensor(const Tensor& value, bool metadata = false);
};

struct Reader {
  const std::vector<uint8_t>& bytes;
  size_t limit;
  size_t offset = 0;

  uint8_t u8() {
    if (offset >= limit) fail("truncated payload");
    return bytes[offset++];
  }
  uint32_t u32() {
    uint32_t value = 0;
    for (int shift = 0; shift < 32; shift += 8) value |= static_cast<uint32_t>(u8()) << shift;
    return value;
  }
  uint64_t u64() {
    uint64_t value = 0;
    for (int shift = 0; shift < 64; shift += 8) value |= static_cast<uint64_t>(u8()) << shift;
    return value;
  }
  int64_t i64() {
    const auto value = u64();
    if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) fail("negative or overflowing integer");
    return static_cast<int64_t>(value);
  }
  int64_t signed64() {
    const auto bits = u64(); int64_t value;
    std::memcpy(&value, &bits, sizeof(value)); return value;
  }
  bool boolean() {
    const auto value = u8();
    if (value > 1) fail("invalid boolean tag");
    return value != 0;
  }
  double f64() {
    const auto bits = u64();
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  std::string string() {
    const auto size = u64();
    if (size > kMaxString || size > limit - offset) fail("invalid string length");
    std::string value(reinterpret_cast<const char*>(bytes.data() + offset), static_cast<size_t>(size));
    offset += static_cast<size_t>(size);
    return value;
  }
  Tensor tensor(bool metadata = false);
};

}  // namespace tide::checkpoint_detail
