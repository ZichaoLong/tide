#include "weighted_region_prepare.h"
extern "C" __global__ __aicore__ void tide_weighted_region_prepare(GM_ADDR received,GM_ADDR caps,
    GM_ADDR scores,GM_ADDR evidence,GM_ADDR current,GM_ADDR ids,GM_ADDR times,GM_ADDR lengths,
    GM_ADDR context_owners,GM_ADDR query_owners,GM_ADDR context_input,GM_ADDR query_input,GM_ADDR statistics,
    GM_ADDR context_rows,GM_ADDR query_rows,GM_ADDR row_errors,GM_ADDR error,int64_t rows,int64_t width,
    int64_t ewidth,int64_t qwidth,int64_t context,int64_t attention,int64_t query_received){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  using namespace tide_weighted_region_device;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)lengths);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;Math math;math.init();
  for(I row=AscendC::GetBlockIdx();row<rows;row+=AscendC::GetBlockNum()){
    auto ctx=(__gm__ float*)context_input+row*(ewidth+2),qry=(__gm__ float*)query_input+row*qwidth;
    auto stats=(__gm__ float*)statistics+row*Statistics;auto e=(__gm__ int32_t*)row_errors+row;*e=0;
    for(I d=0;d<ewidth+2;++d)ctx[d]=0;for(I d=0;d<qwidth;++d)qry[d]=0;
    for(I d=0;d<Statistics;++d)stats[d]=0;
    ((__gm__ I*)context_rows)[row]=-1;((__gm__ I*)query_rows)[row]=-1;
    const I n=((__gm__ I*)lengths)[row];if(n<0||n>width){*e=2;continue;}if(!n)continue;
    if(((__gm__ I*)times)[row]<0){*e=2;continue;}
    auto nodes=(__gm__ I*)ids+row*width;
    for(I j=0;j<n;++j){if(nodes[j]<0)*e=2;for(I k=0;k<j;++k)if(nodes[j]==nodes[k])*e=2;}
    if(*e)continue;
    if(context)((__gm__ I*)context_rows)[row]=((__gm__ I*)context_owners)[row];
    if(attention)((__gm__ I*)query_rows)[row]=((__gm__ I*)query_owners)[row];
    // A present mathematical head cannot be silently omitted with owner=-1.
    if((context&&((__gm__ I*)context_rows)[row]<0)||(attention&&((__gm__ I*)query_rows)[row]<0)){*e=2;continue;}
    *e=prepare(math,(__gm__ float*)received+row*width,(__gm__ float*)caps+row*width,
      (__gm__ float*)scores+row*width,(__gm__ float*)evidence+row*width*ewidth,
      (__gm__ float*)current+row*width*qwidth,n,ewidth,qwidth,context,attention,query_received,ctx,qry,stats);
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
