#pragma once
#include "device_program.h"
#include <memory>
#include <utility>

namespace tide::device_online {
// A fixed-capacity tensor packet between two CANN runtime programs. Append the
// two sides in matching protocol order. Device loops may revisit the same call
// site; notification/storage inventory does not grow with the iteration count.
// The caller decides the common loop/termination protocol on device, launches
// both programs, and drains both before releasing them. No host event polling.
class PeerExchange {
 public:
  using Fields=std::vector<std::pair<at::Tensor,at::Tensor>>; // source,destination
  explicit PeerExchange(Fields,int64_t byte_budget,uint32_t timeout_ms=10000);
  ~PeerExchange();
  PeerExchange(const PeerExchange&)=delete;
  PeerExchange& operator=(const PeerExchange&)=delete;
  void append_send(DeviceProgram&);
  void append_receive(DeviceProgram&);
  int64_t packet_bytes() const;
  // Refuses while any program still retains its commands. A program timeout
  // quarantines those commands, notifications and both endpoints until exit.
  void close();
 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
} // namespace tide::device_online
