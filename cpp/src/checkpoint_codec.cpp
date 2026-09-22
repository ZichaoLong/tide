#include "checkpoint_internal.h"
#include <cstring>
#include <limits>
#include <string_view>

namespace tide::checkpoint_detail {
namespace {
constexpr std::string_view kMagic = "TIDENCK1";
constexpr uint32_t kSchema = 1;
constexpr uint8_t kLittleEndian = 1, kFloat32 = 1, kFloat64 = 2, kCpu = 0;
constexpr uint64_t kMaxString = 1 << 20;
}
bool little_endian() {
  const uint16_t value = 1;
  return *reinterpret_cast<const uint8_t*>(&value) == 1;
}

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
  void tensor(const Tensor& value);
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
  Tensor tensor();
};

uint64_t checksum(const uint8_t* data, size_t size) {
  uint64_t hash = 14695981039346656037ULL;
  for (size_t i = 0; i < size; ++i) {
    hash ^= data[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

uint8_t dtype_code(at::ScalarType dtype) {
  if (dtype == at::kFloat) return kFloat32;
  if (dtype == at::kDouble) return kFloat64;
  fail("only CPU FP32/FP64 tensors are supported");
}

at::ScalarType scalar_type(uint8_t code) {
  if (code == kFloat32) return at::kFloat;
  if (code == kFloat64) return at::kDouble;
  fail("unknown tensor dtype");
}

uint64_t element_count(at::IntArrayRef shape) {
  uint64_t count = 1;
  for (const auto size : shape) {
    if (size < 0 || static_cast<uint64_t>(size) > kMaxFileBytes) fail("invalid tensor shape");
    if (size != 0 && count > kMaxTensorElements / static_cast<uint64_t>(size)) fail("tensor is too large");
    count *= static_cast<uint64_t>(size);
  }
  return count;
}

void Writer::tensor(const Tensor& value) {
  validate_value(value, "save");
  const auto contiguous = value.detach().to(at::kCPU).contiguous();
  if (contiguous.dim() > 64) fail("tensor has too many dimensions");
  u8(kCpu); u8(dtype_code(contiguous.scalar_type())); u8(static_cast<uint8_t>(contiguous.dim()));
  for (const auto size : contiguous.sizes()) i64(size);
  const auto count = element_count(contiguous.sizes());
  u64(count);
  raw(static_cast<const uint8_t*>(contiguous.const_data_ptr()), count * contiguous.element_size());
}

Tensor Reader::tensor() {
  if (u8() != kCpu) fail("tensor device is not CPU");
  const auto dtype = scalar_type(u8());
  const auto dimensions = u8();
  if (dimensions > 64) fail("tensor has too many dimensions");
  std::vector<int64_t> shape;
  shape.reserve(dimensions);
  for (uint8_t i = 0; i < dimensions; ++i) shape.push_back(i64());
  const auto expected = element_count(shape);
  if (u64() != expected) fail("tensor element count mismatch");
  const auto item_size = dtype == at::kFloat ? 4ULL : 8ULL;
  if (expected > (limit - offset) / item_size) fail("truncated tensor payload");
  const auto byte_count = static_cast<size_t>(expected * item_size);
  auto result = at::empty(shape, at::TensorOptions().dtype(dtype).device(at::kCPU));
  if (byte_count) std::memcpy(result.mutable_data_ptr(), bytes.data() + offset, byte_count);
  offset += byte_count;
  validate_value(result, "load");
  return result;
}

void write_group(Writer& writer, const OptimizerGroup& group) {
  if (group.parameters.size() > kMaxCollection) fail("optimizer group is too large");
  writer.u64(group.parameters.size());
  for (const auto& name : group.parameters) writer.string(name);
  writer.f64(group.lr); writer.f64(group.weight_decay); writer.f64(group.momentum);
  writer.f64(group.dampening); writer.f64(group.beta1); writer.f64(group.beta2); writer.f64(group.eps);
  writer.u8(group.nesterov); writer.u8(group.amsgrad); writer.u8(group.maximize);
}

OptimizerGroup read_group(Reader& reader) {
  const auto count = reader.u64();
  if (count > kMaxCollection) fail("optimizer group is too large");
  OptimizerGroup group;
  group.parameters.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i) group.parameters.push_back(reader.string());
  group.lr = reader.f64(); group.weight_decay = reader.f64(); group.momentum = reader.f64();
  group.dampening = reader.f64(); group.beta1 = reader.f64(); group.beta2 = reader.f64(); group.eps = reader.f64();
  group.nesterov = reader.boolean(); group.amsgrad = reader.boolean(); group.maximize = reader.boolean();
  return group;
}

std::vector<uint8_t> encode(const Decoded& record) {
  if (!little_endian()) fail("big-endian hosts are unsupported by schema v1");
  Writer writer;
  writer.bytes.insert(writer.bytes.end(), kMagic.begin(), kMagic.end());
  writer.u32(kSchema); writer.u8(kLittleEndian); writer.string(record.identity);
  const auto& owners = record.owners;
  if (owners.size() > kMaxCollection) fail("too many parameter owners");
  writer.u64(owners.size());
  for (const auto& owner : owners) {
    if (owner.aliases.size() > kMaxCollection) fail("alias partition is too large");
    writer.u64(owner.aliases.size());
    for (const auto& name : owner.aliases) writer.string(name);
    writer.tensor(owner.value);
  }
  writer.u8(record.has_optimizer);
  if (record.has_optimizer) {
    writer.string(record.optimizer_class);
    if (record.groups.size() > kMaxCollection) fail("too many optimizer groups");
    writer.u64(record.groups.size());
    for (const auto& group : record.groups) write_group(writer, group);
    if (record.state.size() > kMaxCollection) fail("optimizer state is too large");
    writer.u64(record.state.size());
    for (const auto& [name, state] : record.state) {
      writer.string(name); writer.i64(state.step);
      uint8_t flags = state.momentum_buffer.defined() | (state.exp_avg.defined() << 1)
                    | (state.exp_avg_sq.defined() << 2) | (state.max_exp_avg_sq.defined() << 3);
      writer.u8(flags);
      if (state.momentum_buffer.defined()) writer.tensor(state.momentum_buffer);
      if (state.exp_avg.defined()) writer.tensor(state.exp_avg);
      if (state.exp_avg_sq.defined()) writer.tensor(state.exp_avg_sq);
      if (state.max_exp_avg_sq.defined()) writer.tensor(state.max_exp_avg_sq);
    }
  }
  const auto hash = checksum(writer.bytes.data(), writer.bytes.size());
  writer.u64(hash);
  return std::move(writer.bytes);
}

Decoded decode(const std::vector<uint8_t>& file) {
  if (!little_endian() || file.size() < kMagic.size() + sizeof(uint32_t) + 1 + sizeof(uint64_t))
    fail("unsupported host or truncated file");
  const size_t payload_size = file.size() - sizeof(uint64_t);
  uint64_t stored = 0;
  std::memcpy(&stored, file.data() + payload_size, sizeof(stored));
  if (checksum(file.data(), payload_size) != stored) fail("checksum mismatch");
  Reader reader{file, payload_size};
  for (const auto expected : kMagic) if (reader.u8() != static_cast<uint8_t>(expected)) fail("magic mismatch");
  if (reader.u32() != kSchema || reader.u8() != kLittleEndian) fail("unsupported schema or byte order");
  Decoded decoded; decoded.identity = reader.string();
  const auto owner_count = reader.u64();
  if (owner_count > kMaxCollection) fail("too many parameter owners");
  decoded.owners.reserve(static_cast<size_t>(owner_count));
  for (uint64_t i = 0; i < owner_count; ++i) {
    const auto alias_count = reader.u64();
    if (alias_count > kMaxCollection) fail("alias partition is too large");
    DecodedOwner owner;
    owner.aliases.reserve(static_cast<size_t>(alias_count));
    for (uint64_t j = 0; j < alias_count; ++j) owner.aliases.push_back(reader.string());
    owner.value = reader.tensor(); decoded.owners.push_back(std::move(owner));
  }
  decoded.has_optimizer = reader.boolean();
  if (decoded.has_optimizer) {
    decoded.optimizer_class = reader.string();
    const auto group_count = reader.u64();
    if (group_count > kMaxCollection) fail("too many optimizer groups");
    decoded.groups.reserve(static_cast<size_t>(group_count));
    for (uint64_t i = 0; i < group_count; ++i) decoded.groups.push_back(read_group(reader));
    const auto state_count = reader.u64();
    if (state_count > kMaxCollection) fail("too many optimizer states");
    for (uint64_t i = 0; i < state_count; ++i) {
      const auto name = reader.string();
      if (decoded.state.count(name)) fail("duplicate optimizer state");
      OptimizerState state; state.step = reader.i64();
      const auto flags = reader.u8();
      if (flags & 1) state.momentum_buffer = reader.tensor();
      if (flags & 2) state.exp_avg = reader.tensor();
      if (flags & 4) state.exp_avg_sq = reader.tensor();
      if (flags & 8) state.max_exp_avg_sq = reader.tensor();
      if (flags & 0xf0) fail("unknown optimizer state slot");
      decoded.state.emplace(name, std::move(state));
    }
  }
  if (reader.offset != payload_size) fail("trailing payload bytes");
  return decoded;
}

}  // namespace tide::checkpoint_detail
