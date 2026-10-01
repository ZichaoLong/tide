#include "kernel_operator.h"
namespace {
using I=int64_t;
__aicore__ inline I find(__gm__ I* table,__gm__ I* events,I buckets,I b,I n,I t) {
  uint64_t x=uint64_t(t)^(uint64_t(n)*0x9e3779b97f4a7c15ULL)^(uint64_t(b)*0xbf58476d1ce4e5b9ULL);
  x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;
  I at=I(x&uint64_t(buckets-1));
  for(I i=0;i<buckets;++i,at=(at+1)&(buckets-1)) {
    const I e=table[at];if(e<0||(events[e*13]==b&&events[e*13+1]==n&&events[e*13+2]==t))return at;
  }
  return -1;
}
}
extern "C" __global__ __aicore__ void tide_event_reverse_links(GM_ADDR events,GM_ADDR event_count,
    GM_ADDR mapping,GM_ADDR config,GM_ADDR windows,GM_ADDR metadata,GM_ADDR count,GM_ADDR lengths,
    GM_ADDR hash,GM_ADDR ranges,GM_ADDR previous,GM_ADDR tails,GM_ADDR initial_lengths,GM_ADDR error,
    int64_t capacity,int64_t cache_rows,int64_t kv_capacity,int64_t nodes,int64_t parameters,int64_t samples,int64_t buckets) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)events);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;if(status[0])return;
  auto e=(__gm__ I*)events,map=(__gm__ I*)mapping,cfg=(__gm__ I*)config,m=(__gm__ I*)metadata;
  auto table=(__gm__ I*)hash,r=(__gm__ I*)ranges,prev=(__gm__ I*)previous,tail=(__gm__ I*)tails;
  auto initial=(__gm__ I*)initial_lengths,final=(__gm__ I*)lengths;
  const I size=((__gm__ I*)event_count)[0],rows=((__gm__ I*)count)[0],owners=samples*parameters;
  if(size<0||size>capacity||rows<0||rows>cache_rows){status[0]=2;return;}
  for(I i=0;i<buckets;++i)table[i]=-1;
  for(I i=0;i<capacity;++i){prev[i]=-1;r[i*4]=-1;r[i*4+1]=0;r[i*4+2]=-1;r[i*4+3]=0;}
  for(I o=0;o<owners;++o){tail[o]=-1;initial[o]=final[o];if(final[o]<0||final[o]>kv_capacity)status[0]=2;}
  for(I i=0;i<size&&!status[0];++i) {
    const I b=e[i*13],n=e[i*13+1],t=e[i*13+2];
    if(b<0||b>=samples||n<0||n>=nodes){status[0]=2;break;}
    const I param=map[n];if(param<0)continue;if(param>=parameters){status[0]=2;break;}
    const I at=find(table,e,buckets,b,n,t),o=b*parameters+param;
    if(at<0||table[at]>=0||(tail[o]>=0&&e[tail[o]*13+2]>=t)){status[0]=2;break;}
    table[at]=i;prev[i]=tail[o];tail[o]=i;
  }
  // Linear expected work: hash each actual KV row, never search all events for
  // every row. Zero-length caches are represented by explicit empty ranges.
  for(I j=0;j<rows&&!status[0];++j) {
    const I b=m[j*5],n=m[j*5+1],t=m[j*5+2],kind=m[j*5+3],row=m[j*5+4];
    if(b<0||b>=samples||n<0||n>=nodes||map[n]<0||kind<0||kind>1||row<0||row>=kv_capacity){status[0]=2;break;}
    const I at=find(table,e,buckets,b,n,t),i=at<0?-1:table[at];if(i<0){status[0]=2;break;}
    auto range=r+i*4+kind*2;
    if(row!=range[1]||(row>0&&range[0]+row!=j)){status[0]=2;break;}
    if(!row)range[0]=j;++range[1];
  }
  for(I o=0;o<owners&&!status[0];++o) {
    const I param=o%parameters;I expected=final[o];
    for(I i=tail[o];i>=0;i=prev[i]) {
      const I old=r[i*4+1],prop=r[i*4+3],window=((__gm__ I*)windows)[param];
      const I want=window>0&&window<old+1?window:old+1;
      const bool active=e[i*13+3],adopt=cfg[param*2]||active,clear=cfg[param*2+1]&&active;
      if(prop!=want||expected!=(adopt?(clear?0:prop):old)){status[0]=2;break;}
      expected=old;
    }
    initial[o]=expected;
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
