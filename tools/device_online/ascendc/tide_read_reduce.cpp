#include "kernel_operator.h"
namespace {using I=int64_t;}
// One serial metadata owner finishes width-tile partials. There are no scalar
// cached stores from competing cores to the same output cache line.
extern "C" __global__ __aicore__ void tide_read_reduce(GM_ADDR fibers,GM_ADDR lengths,
    GM_ADDR kinds,GM_ADDR partials,GM_ADDR scores,GM_ADDR error,int64_t tiles) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)fibers);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  AscendC::TPipe pipe;AscendC::TBuf<AscendC::QuePosition::VECCALC> root_buffer;
  pipe.InitBuffer(root_buffer,32);auto root=root_buffer.Get<float>();
  const I count=((__gm__ I*)lengths)[1];
  auto f=(__gm__ I*)fibers,k=(__gm__ I*)kinds;auto p=(__gm__ float*)partials,d=(__gm__ float*)scores;
  for(I row=0;row<count;++row) {
    float total=p[row*tiles];for(I part=1;part<tiles;++part)total=total+p[row*tiles+part];
    if(k[f[row*4+1]]) {
      root.SetValue(0,total);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      AscendC::Sqrt(root,root,1);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      total=root.GetValue(0);
    }
    d[row]=total; // The following selection preflight rejects nonfinite scores.
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
