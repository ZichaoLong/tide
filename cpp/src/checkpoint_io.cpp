#include "checkpoint_internal.h"
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>

namespace tide::checkpoint_detail {
namespace {
[[noreturn]] void io_error(const char* action, int code = errno) {
  throw std::system_error(code, std::generic_category(), action);
}
class File {
 public:
  explicit File(int fd) : fd_(fd) {}
  ~File() { if (fd_ >= 0) ::close(fd_); }
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  int get() const { return fd_; }
  void close() {
    const int fd = fd_;
    fd_ = -1;  // On Linux close consumes the descriptor even on error.
    if (::close(fd) != 0) io_error("close checkpoint file");
  }
 private:
  int fd_;
};
void write_all(int fd, const uint8_t* data, size_t size) {
  while (size) {
    const auto count = ::write(fd, data, size);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) io_error("write checkpoint");
    if (count == 0) io_error("short checkpoint write", EIO);
    data += count; size -= static_cast<size_t>(count);
  }
}
}  // namespace

std::vector<uint8_t> read_all(const std::filesystem::path& path) {
  File file(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK));
  if (file.get() < 0) io_error("open checkpoint");
  struct stat info;
  if (::fstat(file.get(), &info) != 0) io_error("stat checkpoint");
  if (!S_ISREG(info.st_mode)) fail("checkpoint must be a regular file");
  if (info.st_size < 0 || static_cast<uint64_t>(info.st_size) > kMaxFileBytes)
    fail("checkpoint is too large");
  std::vector<uint8_t> result(static_cast<size_t>(info.st_size));
  size_t offset = 0;
  while (offset < result.size()) {
    const auto count = ::read(file.get(), result.data() + offset, result.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) io_error("read checkpoint");
    if (count == 0) fail("truncated checkpoint file");
    offset += static_cast<size_t>(count);
  }
  uint8_t extra;
  ssize_t tail;
  do { tail = ::read(file.get(), &extra, 1); } while (tail < 0 && errno == EINTR);
  if (tail < 0) io_error("read checkpoint tail");
  if (tail) fail("checkpoint file grew during read");
  file.close();
  return result;
}

void publish_exclusive(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
  const auto parent = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
  if (path.filename().empty()) fail("checkpoint filename is empty");
  File directory(::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (directory.get() < 0) io_error("open checkpoint directory");
  const auto staging = (parent / ("." + path.filename().string() + ".tmp.XXXXXX")).string();
  std::vector<char> mutable_name(staging.begin(), staging.end()); mutable_name.push_back('\0');
  File file(::mkostemp(mutable_name.data(), O_CLOEXEC));
  if (file.get() < 0) io_error("create checkpoint staging file");
  try {
    write_all(file.get(), bytes.data(), bytes.size());
    if (::fsync(file.get()) != 0) io_error("fsync checkpoint staging file");
    file.close();
    if (::link(mutable_name.data(), path.c_str()) != 0) io_error("publish checkpoint without overwrite");
    if (::unlink(mutable_name.data()) != 0) io_error("remove checkpoint staging file");
    if (::fsync(directory.get()) != 0) io_error("fsync checkpoint directory");
    directory.close();
  } catch (...) {
    ::unlink(mutable_name.data());
    throw;  // A post-publication error can leave the complete final file.
  }
}
}  // namespace tide::checkpoint_detail
