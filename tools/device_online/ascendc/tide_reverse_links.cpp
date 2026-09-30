#include "kernel_operator.h"
namespace {
using I=int64_t;
__aicore__ inline uint64_t hash_key(I b,I n,I t) {
  uint64_t x=uint64_t(t)^(uint64_t(n)*0x9e3779b97f4a7c15ULL)^(uint64_t(b)*0xbf58476d1ce4e5b9ULL);
  x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);
}
__aicore__ inline I slot(__gm__ I* table,__gm__ I* events,I buckets,I b,I n,I t) {
  I at=I(hash_key(b,n,t)&uint64_t(buckets-1));
  for(I checked=0;checked<buckets;++checked,at=(at+1)&(buckets-1)) {
    const I row=table[at];if(row<0)return at;
    if(events[row*13]==b&&events[row*13+1]==n&&events[row*13+2]==t)return at;
  }
  return -1;
}
__aicore__ inline I lookup(__gm__ I* table,__gm__ I* events,I buckets,I b,I n,I t) {
  const auto at=slot(table,events,buckets,b,n,t);return at<0?-1:table[at];
}
}
extern "C" __global__ __aicore__ void tide_reverse_links(GM_ADDR metadata,GM_ADDR event_count,
    GM_ADDR fiber_meta,GM_ADDR fiber_count,GM_ADDR pending_meta,GM_ADDR pending_valid,GM_ADDR pending_count,
    GM_ADDR output_meta,GM_ADDR output_valid,GM_ADDR output_count,GM_ADDR sources,GM_ADDR edges,GM_ADDR ports,GM_ADDR hash,
    GM_ADDR messages,GM_ADDR valid,GM_ADDR producer_head,GM_ADDR producer_next,GM_ADDR consumer_head,GM_ADDR consumer_next,
    GM_ADDR scale_head,GM_ADDR scale_next,GM_ADDR stage_offsets,GM_ADDR stages,GM_ADDR error,
    int64_t capacity,int64_t fibers,int64_t pending,int64_t outputs,int64_t nodes,int64_t samples,
    int64_t inputs,int64_t nedges,int64_t nports,int64_t buckets,int64_t cut,int64_t stop) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)metadata);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto e=(__gm__ I*)metadata,h=(__gm__ I*)hash,m=(__gm__ I*)messages,off=(__gm__ I*)stage_offsets;
  auto edge=(__gm__ I*)edges,port=(__gm__ I*)ports,source=(__gm__ I*)sources;
  auto ph=(__gm__ I*)producer_head,pn=(__gm__ I*)producer_next,ch=(__gm__ I*)consumer_head,cn=(__gm__ I*)consumer_next;
  auto sh=(__gm__ I*)scale_head,sn=(__gm__ I*)scale_next,ns=(__gm__ I*)stages;
  auto on=(__gm__ uint8_t*)valid;auto status=(__gm__ int32_t*)error;
  const I events=((__gm__ I*)event_count)[0],atoms=((__gm__ I*)fiber_count)[0],total=fibers+pending+outputs;
  ns[0]=0;
  for(I i=0;i<buckets;++i)h[i]=-1;
  for(I i=0;i<=capacity;++i){off[i]=-1;if(i<capacity){ph[i]=-1;ch[i]=-1;}}
  const I parameters=inputs+2*nedges+nports;
  for(I i=0;i<(parameters?parameters:1);++i)sh[i]=-1;
  for(I i=0;i<total;++i){on[i]=0;pn[i]=-1;sn[2*i]=-1;sn[2*i+1]=-1;for(I j=0;j<4;++j)m[4*i+j]=-1;if(i<fibers)cn[i]=-1;}
  if(status[0]==0&&(events<0||events>capacity||atoms<0||atoms>fibers))status[0]=2;
  for(I i=0;i<events&&status[0]==0;++i) {
    auto row=e+i*13;const I b=row[0],n=row[1],t=row[2],stage=row[12];
    if(b<0||b>=samples||n<0||n>=nodes||t<cut||t>=stop||(row[3]!=0&&row[3]!=1)
        ||stage<0||stage>=capacity||(!i&&stage!=0)||(i&&(stage<e[(i-1)*13+12]||stage>e[(i-1)*13+12]+1))) {status[0]=2;break;}
    const I place=slot(h,e,buckets,b,n,t);
    if(place<0||h[place]>=0){status[0]=2;break;}h[place]=i;
    if(!i||stage!=e[(i-1)*13+12])off[stage]=i;
    ns[0]=stage+1;
  }
  if(status[0]==0)off[ns[0]]=events;
  I seen_pending=0,seen_outputs=0;
  for(I i=0;i<total&&status[0]==0;++i) {
    const bool fiber=i<fibers,is_pending=i>=fibers&&i<fibers+pending;
    const I local=fiber?i:is_pending?i-fibers:i-fibers-pending;
    const bool live=fiber?i<atoms:is_pending?((__gm__ uint8_t*)pending_valid)[local]:((__gm__ uint8_t*)output_valid)[local];
    if(!live)continue;
    auto coords=(__gm__ I*)(fiber?fiber_meta:is_pending?pending_meta:output_meta)+local*6;
    const I b=coords[0],n=coords[1],t=coords[2],kind=coords[3],id=coords[4],position=coords[5];
    if(b<0||b>=samples||n<0||n>=nodes||t<0||position<0){status[0]=2;break;}
    I consumer=-1,producer=-1,aggregate=-1,delivery=-1;
    if(fiber||is_pending) {
      if((kind!=0&&kind!=1)||id<0||id>=(kind==0?inputs:nedges)
          ||source[(id+(kind==0?0:inputs))*2]!=n){status[0]=2;break;}
      if(fiber) {
        consumer=lookup(h,e,buckets,b,n,t);if(consumer<0){status[0]=2;break;}
        aggregate=id+(kind==0?0:inputs);
      } else {++seen_pending;if(t<stop){status[0]=2;break;}}
      if(kind==1) {
        const I src=edge[id*4],delay=edge[id*4+2];
        if(src<0||src>=nodes||edge[id*4+1]!=n||delay<=0){status[0]=2;break;}
        // A producer before the complete cut belongs to the incoming graph.
        // Keep its message gradient as a boundary input, not a current parameter gradient.
        if(t>=delay&&t-delay>=cut) {
          producer=lookup(h,e,buckets,b,src,t-delay);
          if(producer<0||!e[producer*13+3]||position!=t-delay
              ||(consumer>=0&&e[producer*13+12]>=e[consumer*13+12])){status[0]=2;break;}
          delivery=inputs+nedges+edge[id*4+3];
        }
      }
    } else {
      ++seen_outputs;
      if(kind!=0||id<0||id>=nports||port[id*2]!=n||position!=t){status[0]=2;break;}
      producer=lookup(h,e,buckets,b,n,t);
      if(producer<0||!e[producer*13+3]){status[0]=2;break;}
      delivery=inputs+nedges+port[id*2+1];
    }
    m[i*4]=consumer;m[i*4+1]=producer;m[i*4+2]=aggregate;m[i*4+3]=delivery;on[i]=1;
  }
  if(status[0]==0&&(seen_pending!=((__gm__ I*)pending_count)[0]||seen_outputs!=((__gm__ I*)output_count)[0]))status[0]=2;
  // Reverse insertion yields stable ascending physical-message order. Each
  // destination tile can later gather its contributors without scatter races.
  if(status[0]==0)for(I i=total;i>0;) {--i;if(!on[i])continue;
    const I consumer=m[i*4],producer=m[i*4+1];
    if(producer>=0){pn[i]=ph[producer];ph[producer]=i;}
    if(consumer>=0){cn[i]=ch[consumer];ch[consumer]=i;}
    for(I j=0;j<2;++j){const I param=m[i*4+2+j];if(param>=0){sn[2*i+j]=sh[param];sh[param]=2*i+j;}}
  }
  if(status[0]){ns[0]=0;for(I i=0;i<total;++i)on[i]=0;}
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
