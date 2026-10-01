#pragma once
#include "sharded_parameter_banks.h"
#include "sharded_parameter_sources.h"
#include "peer_exchange.h"
namespace tide::device_online {
// Each field describes one already-computed source and disjoint local writes.
// Reduction callers submit contribution ordinal k before k+1 for every owner.
struct OwnerStreamField {
  ParameterContribution source;
  std::vector<ParameterDestination> destinations;
  at::Tensor connected; // One destination owner flag; undefined for publication.
};
struct OwnerStream {
  std::unique_ptr<PeerExchange> peer;
  int64_t reserved_bytes=0,capacity=0,iterations=0;
};
// Descriptor/counter allowance excluding the float packet, at both endpoints.
int64_t owner_stream_metadata_bytes(int64_t fields,int64_t writes,bool peer);
// Fixed tensor geometry only. Payload/connection flags stay on device. Packet
// buffers are reused by a CANN loop, including the final incomplete chunk.
OwnerStream append_owner_stream(CannProgram& source,CannProgram& destination,
    const std::vector<OwnerStreamField>&,const at::Tensor& source_error,const at::Tensor& destination_error,
    int64_t tensor_budget,bool accumulate);
} // namespace tide::device_online
