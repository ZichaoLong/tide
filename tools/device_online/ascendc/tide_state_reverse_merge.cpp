#include "fiber_vector.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_state_reverse_merge(GM_ADDR count,GM_ADDR destinations,GM_ADDR ids,
    GM_ADDR content,GM_ADDR content_on,GM_ADDR initial,GM_ADDR initial_on,GM_ADDR out_content,GM_ADDR out_on,
    GM_ADDR out_initial,GM_ADDR out_initial_on,GM_ADDR local_error,GM_ADDR error,
    int64_t capacity,int64_t global_capacity,int64_t nodes,int64_t global_nodes,int64_t samples,int64_t width,int64_t phase) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)count);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;
  if(phase==4){if(AscendC::GetBlockIdx()==0&&!status[0])status[0]=((__gm__ int32_t*)local_error)[0];
    AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);return;}
  auto dst=(__gm__ I*)destinations,id=(__gm__ I*)ids;const I size=((__gm__ I*)count)[0];
  auto on=(__gm__ uint8_t*)content_on,carry=(__gm__ uint8_t*)initial_on;
  if(phase==0||phase==2) {
    if(AscendC::GetBlockIdx()!=0)return;if(!status[0])status[0]=((__gm__ int32_t*)local_error)[0];
    if(!status[0]&&(size<0||size>capacity))status[0]=2;
    for(I i=0;i<size&&!status[0];++i)if(dst[i]<0||dst[i]>=global_capacity)status[0]=2;
    if(phase==0)for(I n=0;n<nodes&&!status[0];++n)if(id[n]<0||id[n]>=global_nodes)status[0]=2;
    if(!status[0]) {
      for(I i=0;i<size;++i)if(phase==0||on[i])((__gm__ uint8_t*)out_on)[dst[i]]=on[i];
      if(phase==0)for(I b=0;b<samples;++b)for(I n=0;n<nodes;++n)((__gm__ uint8_t*)out_initial_on)[b*global_nodes+id[n]]=carry[b*nodes+n];
    }
    AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);return;
  }
  if(status[0])return;const I tiles=(width+255)/256,rows=size+(phase==1?samples*nodes:0);
  tide_device::FiberVector op;op.init();auto x=op.x(),y=op.y();
  for(I task=AscendC::GetBlockIdx();task<rows*tiles;task+=AscendC::GetBlockNum()) {
    const I row=task/tiles,start=(task%tiles)*256;const uint32_t length=width-start<256?width-start:256;
    if(row<size) {
      if(!on[row])continue;op.load(x,(__gm__ float*)content,row*width+start,length);
      if(phase==3){op.load(y,(__gm__ float*)out_content,dst[row]*width+start,length);AscendC::Add(x,x,y,length);AscendC::PipeBarrier<PIPE_V>();}
      op.save(x,(__gm__ float*)out_content,dst[row]*width+start,length);
      if(phase==3){op.load(x,(__gm__ float*)initial,row*width+start,length);op.load(y,(__gm__ float*)out_initial,dst[row]*width+start,length);
        AscendC::Add(x,x,y,length);AscendC::PipeBarrier<PIPE_V>();op.save(x,(__gm__ float*)out_initial,dst[row]*width+start,length);}
    } else {const I key=row-size;if(!carry[key])continue;op.load(x,(__gm__ float*)initial,key*width+start,length);
      op.save(x,(__gm__ float*)out_initial,(key/nodes*global_nodes+id[key%nodes])*width+start,length);}
  }
}
