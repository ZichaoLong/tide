#include "kernel_operator.h"
namespace {
using I=int64_t;
__aicore__ inline uint64_t hash_key(I b,I r,I t) {
  uint64_t x=uint64_t(t)^(uint64_t(r)*0x9e3779b97f4a7c15ULL)^(uint64_t(b)*0xbf58476d1ce4e5b9ULL);
  x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;return x^(x>>31);
}
}
extern "C" __global__ __aicore__ void tide_control_plan(GM_ADDR metadata,GM_ADDR values,GM_ADDR count,GM_ADDR range,
    GM_ADDR config,GM_ADDR connected,GM_ADDR hash,GM_ADDR frames,GM_ADDR next,GM_ADDR owner_head,GM_ADDR owner_next,
    GM_ADDR probability_gradient,GM_ADDR scores,GM_ADDR read_connected,GM_ADDR cot_connected,GM_ADDR parameters,GM_ADDR error,
    int64_t capacity,int64_t width,int64_t nodes,int64_t samples,int64_t buckets,int64_t mode,int64_t emit_mode,float zeta) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)metadata);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;if(status[0])return;
  auto e=(__gm__ I*)metadata,cfg=(__gm__ I*)config,h=(__gm__ I*)hash,f=(__gm__ I*)frames,nx=(__gm__ I*)next;
  auto oh=(__gm__ I*)owner_head,ox=(__gm__ I*)owner_next;
  auto on=(__gm__ uint8_t*)connected,rc=(__gm__ uint8_t*)read_connected,cc=(__gm__ uint8_t*)cot_connected;
  auto v=(__gm__ float*)values,dp=(__gm__ float*)probability_gradient,ds=(__gm__ float*)scores;
  const I size=((__gm__ I*)count)[0],first=((__gm__ I*)range)[0],stride=5*width+2;
  if(size<0||size>capacity||first<0||first>capacity-size){status[0]=2;return;}
  if(mode==0) {
    for(I i=0;i<buckets;++i)h[i]=-1;
    for(I n=0;n<nodes;++n)oh[n]=-1;
    // Validate all rows before publishing any structural connections.
    for(I i=0;i<size;++i) {
      auto row=e+(first+i)*13;const I n=row[1];
      if(row[0]<0||row[0]>=samples||n<0||n>=nodes||row[2]<0||(row[3]!=0&&row[3]!=1)||(on[i]&&!row[3])){status[0]=2;return;}
    }
    for(I i=size;i>0;) {
      --i;auto row=e+(first+i)*13;const I b=row[0],n=row[1],r=cfg[n*3],t=row[2];
      I at=I(hash_key(b,r,t)&uint64_t(buckets-1));
      for(I checked=0;checked<buckets;++checked,at=(at+1)&(buckets-1)) {
        const I j=h[at];if(j<0)break;
        auto other=e+(first+j)*13;if(other[0]==b&&cfg[other[1]*3]==r&&other[2]==t)break;
      }
      nx[i]=h[at];h[at]=i;f[i]=at;ox[i]=oh[n];oh[n]=i;
    }
    for(I at=0;at<buckets;++at)if(h[at]>=0) {
      bool live=false;for(I i=h[at];i>=0;i=nx[i])live|=on[i]&&cfg[e[(first+i)*13+1]*3+1]>=0;
      for(I i=h[at];i>=0;i=nx[i]) {
        const I n=e[(first+i)*13+1],read=cfg[n*3+1];
        if(on[i]&&read>=0)cc[i*5]=1; // HST's declared zero is connected.
        if(live&&read>=0){rc[i]=1;cc[i*5+read]=1;if(!cfg[n*3+2])((__gm__ uint8_t*)parameters)[n]=1;}
      }
    }
  } else {
    for(I i=0;i<size;++i)if(emit_mode==1)dp[i]=dp[i]*zeta;
    for(I at=0;at<buckets;++at)if(h[at]>=0) {
      float center=0.f;
      for(I i=h[at];i>=0;i=nx[i])center=center+dp[i]*v[(first+i)*stride+5*width+1];
      for(I i=h[at];i>=0;i=nx[i])if(rc[i])ds[i]=v[(first+i)*stride+5*width+1]*(dp[i]-center);
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
