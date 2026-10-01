#include "kernel_operator.h"
namespace {
using I=int64_t;
__aicore__ inline uint64_t hash_key(I b,I n,I t,I s) {
  uint64_t x=uint64_t(t)^(uint64_t(n)*0x9e3779b97f4a7c15ULL)^(uint64_t(b)*0xbf58476d1ce4e5b9ULL)
    ^(uint64_t(s)*0x94d049bb133111ebULL);
  x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);
}
__aicore__ inline I slot(__gm__ I* hash,__gm__ I* meta,I buckets,I b,I n,I t,I s) {
  I at=I(hash_key(b,n,t,s)&uint64_t(buckets-1));
  for(I i=0;i<buckets;++i,at=(at+1)&(buckets-1)) {
    const I row=hash[at];if(row<0)return at;
    if(meta[row*6]==b&&meta[row*6+1]==n&&meta[row*6+2]==t&&meta[row*6+4]==s)return at;
  }
  return -1;
}
}
extern "C" __global__ __aicore__ void tide_emission_reverse_links(GM_ADDR events,GM_ADDR event_count,
    GM_ADDR emission,GM_ADDR emission_count,GM_ADDR messages,GM_ADDR valid,GM_ADDR offsets,GM_ADDR mapping,
    GM_ADDR hash,GM_ADDR rows,GM_ADDR parameters,GM_ADDR error,int64_t capacity,int64_t emissions,
    int64_t total,int64_t nodes,int64_t samples,int64_t scale_offset,int64_t buckets,int64_t projections) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)events);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto e=(__gm__ I*)events,m=(__gm__ I*)emission,msg=(__gm__ I*)messages,off=(__gm__ I*)offsets;
  auto map=(__gm__ I*)mapping,h=(__gm__ I*)hash,r=(__gm__ I*)rows,p=(__gm__ I*)parameters;
  auto status=(__gm__ int32_t*)error;
  const I count=((__gm__ I*)emission_count)[0],ne=((__gm__ I*)event_count)[0];
  for(I i=0;i<buckets;++i)h[i]=-1;
  for(I i=0;i<total;++i){r[i]=-1;p[i]=-1;}
  if(status[0]==0&&(count<0||count>emissions||ne<0||ne>capacity))status[0]=2;
  for(I i=0;i<count&&status[0]==0;++i) {
    const I b=m[i*6],n=m[i*6+1],t=m[i*6+2],s=m[i*6+4],param=m[i*6+5];
    if(b<0||b>=samples||n<0||n>=nodes||t<0||m[i*6+3]!=t||s<0||s>=off[n+1]-off[n]
        ||param!=map[off[n]+s]||param< -1||param>=projections){status[0]=2;break;}
    const I at=slot(h,m,buckets,b,n,t,s);
    if(at<0||h[at]>=0){status[0]=2;break;}h[at]=i;
  }
  for(I i=0;i<total&&status[0]==0;++i)if(((__gm__ uint8_t*)valid)[i]&&msg[i*4+1]>=0) {
    const I event=msg[i*4+1];if(event>=ne){status[0]=2;break;}
    const I b=e[event*13],n=e[event*13+1],t=e[event*13+2],s=msg[i*4+3]-scale_offset-off[n];
    if(s<0||s>=off[n+1]-off[n]){status[0]=2;break;}
    const I at=slot(h,m,buckets,b,n,t,s),row=at<0?-1:h[at];
    if(row<0){status[0]=2;break;}r[i]=row;p[i]=m[row*6+5];
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
