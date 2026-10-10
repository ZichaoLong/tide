#include "peer_kernel.h"
#include <cuda/atomic>

namespace tide::device_online::cuda_backend {
namespace {
using Atomic=cuda::atomic_ref<uint64_t,cuda::thread_scope_system>;
__device__ uint64_t now(){uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__device__ void failed(){asm volatile("trap;");}
__device__ void deadline(uint64_t start,uint64_t timeout){if(now()-start>=timeout)failed();__nanosleep(256);}
__global__ void send(uint64_t* flags,uint64_t timeout) {
  Atomic ready(flags[0]),consumed(flags[1]);
  auto previous=ready.load(cuda::memory_order_relaxed);
  if(previous==UINT64_MAX||consumed.load(cuda::memory_order_acquire)!=previous){failed();return;}
  ready.store(previous+1,cuda::memory_order_release);
  const auto start=now();
  while(consumed.load(cuda::memory_order_acquire)!=previous+1)deadline(start,timeout);
}
__global__ void receive(uint64_t* flags,const PeerField* fields,int64_t count,uint64_t timeout) {
  __shared__ uint64_t generation;
  if(threadIdx.x==0) {
    Atomic ready(flags[0]),consumed(flags[1]);
    const auto previous=consumed.load(cuda::memory_order_relaxed),start=now();
    uint64_t next;
    do{next=ready.load(cuda::memory_order_acquire);if(next==previous)deadline(start,timeout);}while(next==previous);
    if(previous==UINT64_MAX||next!=previous+1){failed();return;}
    generation=next;
  }
  __syncthreads();
  for(int64_t f=0;f<count;++f)for(uint64_t i=threadIdx.x;i<fields[f].bytes;i+=blockDim.x)
    fields[f].destination[i]=fields[f].source[i];
  // Every copying thread publishes its stores before thread0 acknowledges.
  __threadfence_system();__syncthreads();
  if(threadIdx.x==0)Atomic(flags[1]).store(generation,cuda::memory_order_release);
}
}
void peer_send(cudaStream_t s,uint64_t* flags,uint64_t timeout){send<<<1,1,0,s>>>(flags,timeout);}
void peer_receive(cudaStream_t s,uint64_t* flags,const PeerField* fields,int64_t n,uint64_t timeout){receive<<<1,256,0,s>>>(flags,fields,n,timeout);}
}
