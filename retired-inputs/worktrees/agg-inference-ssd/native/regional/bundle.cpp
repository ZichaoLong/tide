#include "bundle.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>

namespace tide_native {
namespace {
const char* magic = "tide-regional-tensors-v1 little-endian float32";
void little_endian() {
  const uint32_t one = 1;
  TORCH_CHECK(*reinterpret_cast<const uint8_t*>(&one) == 1,
              "this implementation requires a little-endian host");
}
std::string digest(const char* bytes, size_t length) {
  uint64_t value = UINT64_C(14695981039346656037);
  for (size_t i = 0; i < length; ++i) value = (value ^ static_cast<uint8_t>(bytes[i])) * UINT64_C(1099511628211);
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << value;
  return stream.str();
}
void sync_file(const std::filesystem::path& path, bool directory = false) {
  int fd = ::open(path.c_str(), O_RDONLY | (directory ? O_DIRECTORY : 0));
  TORCH_CHECK(fd >= 0, "open for fsync failed: ", path.string());
  const int result = ::fsync(fd);
  ::close(fd);
  TORCH_CHECK(result == 0, "fsync failed: ", path.string());
}
}  // namespace
void atomic_text(const std::filesystem::path& path, const std::string& value) {
  auto stage = path.string() + ".staging";
  {
    std::ofstream out(stage, std::ios::binary | std::ios::trunc);
    out << value;
    out.flush();
    TORCH_CHECK(out.good(), "write failed: ", stage);
  }
  publish_staged(stage, path);
}
void publish_staged(const std::filesystem::path& stage, const std::filesystem::path& path) {
  sync_file(stage);
  std::filesystem::rename(stage, path);
  sync_file(path.parent_path(), true);
}
std::string json_string(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char ch : value) {
    if (ch == '"' || ch == '\\') out << '\\' << ch;
    else if (ch < 32) out << "\\u" << std::hex << std::setfill('0') << std::setw(4) << int(ch);
    else out << ch;
  }
  out << '"';
  return out.str();
}
Tensors read_bundle(const std::filesystem::path& root) {
  little_endian();
  std::ifstream index(root / "index.txt");
  std::string line;
  TORCH_CHECK(bool(std::getline(index, line)) && line == magic, "bad tensor bundle header");
  Tensors result;
  while (std::getline(index, line)) {
    std::istringstream row(line);
    std::string name, hash, extra;
    int rank = -1;
    TORCH_CHECK(bool(row >> name >> rank) && rank >= 0 && rank <= 8,
                "invalid tensor name or rank");
    TORCH_CHECK(name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.") == std::string::npos,
                "invalid tensor name");
    std::vector<int64_t> shape(rank);
    int64_t count = 1;
    for (auto& dim : shape) {
      TORCH_CHECK(bool(row >> dim) && dim > 0 && dim <= 1000000000 / count,
                  "invalid or oversized tensor shape");
      count *= dim;
    }
    TORCH_CHECK(bool(row >> hash) && !(row >> extra) && hash.size() == 16,
                "invalid tensor index row");
    auto path = root / (name + ".f32");
    TORCH_CHECK(std::filesystem::file_size(path) == static_cast<uint64_t>(count) * 4,
                "tensor byte count mismatch: ", name);
    auto value = at::empty(shape, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(value.data_ptr<float>()), count * 4);
    TORCH_CHECK(input.good() && digest(reinterpret_cast<char*>(value.data_ptr<float>()), count * 4) == hash,
                "tensor integrity failure: ", name);
    TORCH_CHECK(at::isfinite(value).all().item<bool>(), "nonfinite fixture tensor: ", name);
    TORCH_CHECK(result.emplace(name, value).second, "duplicate tensor: ", name);
  }
  TORCH_CHECK(index.eof() && !result.empty(), "incomplete or empty tensor index");
  return result;
}
void write_bundle(const std::filesystem::path& root, const Tensors& values) {
  little_endian();
  TORCH_CHECK(std::filesystem::create_directory(root), "tensor output directory must be new");
  std::ostringstream index;
  index << magic << '\n';
  for (const auto& item : values) {
    auto value = item.second.detach().to(at::kCPU).to(at::kFloat).contiguous();
    TORCH_CHECK(at::isfinite(value).all().item<bool>(), "nonfinite result tensor: ", item.first);
    auto path = root / (item.first + ".f32");
    auto bytes = reinterpret_cast<const char*>(value.data_ptr<float>());
    const auto size = value.numel() * 4;
    {
      std::ofstream out(path, std::ios::binary);
      out.write(bytes, size);
      out.flush();
      TORCH_CHECK(out.good(), "tensor write failed");
    }
    sync_file(path);
    index << item.first << ' ' << value.dim();
    for (const auto dim : value.sizes()) index << ' ' << dim;
    index << ' ' << digest(bytes, size) << '\n';
  }
  atomic_text(root / "index.txt", index.str());
}
}  // namespace tide_native
