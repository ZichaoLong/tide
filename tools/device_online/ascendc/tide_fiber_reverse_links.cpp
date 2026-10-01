#include "kernel_operator.h"
#include "state_clock.h"
namespace {
using I=int64_t;
__aicore__ inline I find(__gm__ I* table,__gm__ I* events,I buckets,I b,I n,I t) {
  uint64_t x=uint64_t(t)^(uint64_t(n)*0x9e3779b97f4a7c15ULL)^(uint64_t(b)*0xbf58476d1ce4e5b9ULL);
  x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;I at=I(x&uint64_t(buckets-1));
  for(I i=0;i<buckets;++i,at=(at+1)&(buckets-1)) {
    const I e=table[at];if(e<0||(events[e*13]==b&&events[e*13+1]==n&&events[e*13+2]==t))return at;
  }
  return -1;
}
__aicore__ inline I slot(__gm__ I* meta,__gm__ I* sources,I row,I inputs) {
  const I source=meta[row*6+4]+(meta[row*6+3]?inputs:0);return sources[source*2+1];
}
}
extern "C" __global__ __aicore__ void tide_fiber_reverse_links(GM_ADDR events,GM_ADDR event_count,
    GM_ADDR mapping,GM_ADDR config,GM_ADDR metadata,GM_ADDR cache_count,GM_ADDR lengths,GM_ADDR clock,
    GM_ADDR consumer_head,GM_ADDR consumer_next,GM_ADDR fiber_meta,GM_ADDR sources,
    GM_ADDR hash,GM_ADDR ranges,GM_ADDR previous,GM_ADDR tails,GM_ADDR initial_lengths,
    GM_ADDR tokens,GM_ADDR scratch,GM_ADDR ticks,GM_ADDR error,int64_t capacity,int64_t cache_rows,
    int64_t kv_capacity,int64_t nodes,int64_t parameters,int64_t samples,int64_t buckets,
    int64_t fibers,int64_t inputs,int64_t physical_sources,int64_t max_ticks) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)events);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;if(status[0])return;
  auto e=(__gm__ I*)events,map=(__gm__ I*)mapping,cfg=(__gm__ I*)config,m=(__gm__ I*)metadata;
  auto table=(__gm__ I*)hash,r=(__gm__ I*)ranges,prev=(__gm__ I*)previous,tail=(__gm__ I*)tails;
  auto initial=(__gm__ I*)initial_lengths,final=(__gm__ I*)lengths,dt=(__gm__ I*)ticks;
  auto ch=(__gm__ I*)consumer_head,cn=(__gm__ I*)consumer_next,f=(__gm__ I*)fiber_meta,s=(__gm__ I*)sources;
  auto rows=(__gm__ I*)tokens,tmp=(__gm__ I*)scratch;
  const I size=((__gm__ I*)event_count)[0],nr=((__gm__ I*)cache_count)[0],owners=samples*parameters;
  if(size<0||size>capacity||nr<0||nr>cache_rows){status[0]=2;return;}
  for(I i=0;i<buckets;++i)table[i]=-1;
  for(I i=0;i<capacity;++i){prev[i]=-1;dt[i]=0;for(I j=0;j<6;++j)r[i*6+j]=j==0||j==2?-1:0;}
  for(I owner=0;owner<owners;++owner){tail[owner]=-1;initial[owner]=final[owner];if(final[owner]<0||final[owner]>kv_capacity)status[0]=2;}
  I offset=0;
  for(I i=0;i<size&&!status[0];++i) {
    const I b=e[i*13],n=e[i*13+1],time=e[i*13+2];
    if(b<0||b>=samples||n<0||n>=nodes){status[0]=2;break;}
    const I parameter=map[n];if(parameter<0)continue;if(parameter>=parameters){status[0]=2;break;}
    const I owner=b*parameters+parameter,at=find(table,e,buckets,b,n,time);
    if(at<0||table[at]>=0||(tail[owner]>=0&&e[tail[owner]*13+2]>=time)){status[0]=2;break;}
    table[at]=i;prev[i]=tail[owner];tail[owner]=i;r[i*6+4]=offset;
    for(I row=ch[i];row>=0;row=cn[row]) {
      if(row>=fibers||offset>=fibers){status[0]=2;break;}
      const I source=f[row*6+4]+(f[row*6+3]?inputs:0);
      if(source<0||source>=physical_sources){status[0]=2;break;}
      rows[offset++]=row;
    }
    const I first=r[i*6+4],count=offset-first;r[i*6+5]=count;
    if(count<1){status[0]=2;break;}
    // Stable merge sort of actual sources by logical slot, O(F log F), with
    // reusable integer scratch. Physical message identities are never merged.
    for(I step=1;step<count;step*=2) {
      for(I start=first;start<offset;start+=step*2) {
        I middle=start+step;if(middle>offset)middle=offset;
        I end=start+step*2;if(end>offset)end=offset;I left=start,right=middle;
        for(I j=start;j<end;++j) {
          const bool take_left=left<middle&&(right>=end||slot(f,s,rows[left],inputs)<=slot(f,s,rows[right],inputs));
          tmp[j]=take_left?rows[left++]:rows[right++];
        }
      }
      for(I j=first;j<offset;++j)rows[j]=tmp[j];
    }
    for(I j=first+1;j<offset;++j)if(slot(f,s,rows[j-1],inputs)>=slot(f,s,rows[j],inputs)){status[0]=2;break;}
  }
  for(I j=0;j<nr&&!status[0];++j) {
    const I b=m[j*5],n=m[j*5+1],time=m[j*5+2],kind=m[j*5+3],row=m[j*5+4];
    if(b<0||b>=samples||n<0||n>=nodes||kind<0||kind>1||row<0||row>=kv_capacity){status[0]=2;break;}
    if(map[n]<0)continue;const I at=find(table,e,buckets,b,n,time),i=at<0?-1:table[at];if(i<0){status[0]=2;break;}
    auto range=r+i*6+kind*2;
    if(row!=range[1]||(row>0&&range[0]+row!=j)){status[0]=2;break;}
    if(!row)range[0]=j;++range[1];
  }
  for(I owner=0;owner<owners&&!status[0];++owner) {
    const I param=owner%parameters;I expected=final[owner];
    for(I i=tail[owner];i>=0;i=prev[i]) {
      const I old=r[i*6+1],proposal=r[i*6+3],n=e[i*13+1];
      const bool active=e[i*13+3],adopt=cfg[param*2]||active,clear=cfg[param*2+1]&&active;
      if(proposal!=old+r[i*6+5]||expected!=(adopt?(clear?0:proposal):old)){status[0]=2;break;}
      if(old) {
        I local,previous_time;
        if(!tide_device::local_time(e[i*13+2],(__gm__ I*)clock,n,local)
            ||!tide_device::local_time(e[i*13+4],(__gm__ I*)clock,n,previous_time)){status[0]=9;break;}
        const uint64_t gap=(uint64_t(local)+1)-uint64_t(previous_time+1);
        if(gap>uint64_t(max_ticks)){status[0]=8;break;}dt[i]=I(gap);
      }
      expected=old;
    }
    initial[owner]=expected;
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
