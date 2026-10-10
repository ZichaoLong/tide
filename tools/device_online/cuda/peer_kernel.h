#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
namespace tide::device_online::cuda_backend {
struct PeerField {const uint8_t* source;uint8_t* destination;uint64_t bytes;};
void peer_send(cudaStream_t,uint64_t* flags,uint64_t timeout_ns);
void peer_receive(cudaStream_t,uint64_t* flags,const PeerField*,int64_t count,uint64_t timeout_ns);
}
