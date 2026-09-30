#include "kernel_operator.h"
namespace {
using I=int64_t;
__aicore__ inline I field(I row,I j,I fibers,__gm__ I* fm,__gm__ I* pm) {
  return row<fibers?fm[row*6+j]:pm[(row-fibers)*6+j];
}
__aicore__ inline uint64_t mix(uint64_t x) {
  x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);
}
__aicore__ inline I slot(__gm__ I* hash,__gm__ I* fm,__gm__ I* pm,I fibers,I buckets,__gm__ I* key) {
  uint64_t value=0x9e3779b97f4a7c15ULL;for(I j=0;j<6;++j)value=mix(value^uint64_t(key[j]));
  I at=I(value&uint64_t(buckets-1));
  for(I checked=0;checked<buckets;++checked,at=(at+1)&(buckets-1)) {
    const I row=hash[at];if(row<0)return at;bool same=true;
    for(I j=0;j<6;++j)same&=field(row,j,fibers,fm,pm)==key[j];if(same)return at;
  }
  return -1;
}
}
extern "C" __global__ __aicore__ void tide_window_bridge_meta(GM_ADDR earlier_meta,GM_ADDR earlier_valid,
    GM_ADDR fiber_meta,GM_ADDR pending_meta,GM_ADDR links,GM_ADDR valid,GM_ADDR message_connected,GM_ADDR initial_connected,
    GM_ADDR local_pending_connected,GM_ADDR local_final_connected,GM_ADDR hash,GM_ADDR mapping,GM_ADDR pending_connected,GM_ADDR final_connected,
    GM_ADDR error,int64_t rows,int64_t fibers,int64_t pending,int64_t states,int64_t buckets) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)hash);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto h=(__gm__ I*)hash,map=(__gm__ I*)mapping,fm=(__gm__ I*)fiber_meta,pm=(__gm__ I*)pending_meta,em=(__gm__ I*)earlier_meta;
  auto pc=(__gm__ uint8_t*)pending_connected,fc=(__gm__ uint8_t*)final_connected;auto status=(__gm__ int32_t*)error;
  for(I i=0;i<rows;++i){map[i]=-1;pc[i]=0;}for(I i=0;i<states;++i)fc[i]=0;
  if(status[0]==0) {
    for(I i=0;i<buckets;++i)h[i]=-1;
    for(I i=0;i<fibers+pending;++i)if(((__gm__ uint8_t*)valid)[i]&&((__gm__ I*)links)[i*4+1]<0) {
      auto key=i<fibers?fm+i*6:pm+(i-fibers)*6;const I at=slot(h,fm,pm,fibers,buckets,key);
      if(at<0||h[at]>=0){status[0]=22;break;}h[at]=i;
    }
    for(I i=0;i<rows&&status[0]==0;++i)if(((__gm__ uint8_t*)earlier_valid)[i]) {
      const I at=slot(h,fm,pm,fibers,buckets,em+i*6);if(at<0||h[at]<0){status[0]=22;break;}
      map[i]=h[at];pc[i]=((__gm__ uint8_t*)local_pending_connected)[i]||((__gm__ uint8_t*)message_connected)[map[i]];
    }
    if(status[0]==0)for(I i=0;i<states;++i)fc[i]=((__gm__ uint8_t*)local_final_connected)[i]||((__gm__ uint8_t*)initial_connected)[i];
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
