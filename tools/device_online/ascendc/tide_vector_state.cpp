#include "kernel_operator.h"
#include "state_clock.h"
namespace {
using I=int64_t;
constexpr uint32_t tile=256;
class VectorState {
 public:
  __aicore__ inline void run(GM_ADDR lengths,GM_ADDR content,GM_ADDR config,GM_ADDR coefficients,
      GM_ADDR retention,GM_ADDR policy,GM_ADDR state,GM_ADDR comparisons,GM_ADDR metadata,GM_ADDR diagnostics,
      GM_ADDR proposals,GM_ADDR error,I width,I nodes,bool trace) {
    AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)metadata);
    AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
    if(((__gm__ int32_t*)error)[0])return;
    auto events=(__gm__ I*)metadata,cfg=(__gm__ I*)config;
    auto rho=(__gm__ float*)retention;
    AscendC::GlobalTensor<float> h,a,s,c,v,ap;
    h.SetGlobalBuffer((__gm__ float*)content);a.SetGlobalBuffer((__gm__ float*)coefficients);
    s.SetGlobalBuffer((__gm__ float*)state);c.SetGlobalBuffer((__gm__ float*)comparisons);
    v.SetGlobalBuffer((__gm__ float*)diagnostics);ap.SetGlobalBuffer((__gm__ float*)proposals);
    pipe.InitBuffer(input,1,tile*sizeof(float));pipe.InitBuffer(output,1,tile*sizeof(float));
    pipe.InitBuffer(old_buffer,tile*sizeof(float));pipe.InitBuffer(proposal_buffer,tile*sizeof(float));
    pipe.InitBuffer(summary_buffer,tile*sizeof(float));pipe.InitBuffer(coefficient_buffer,tile*sizeof(float));
    auto old=old_buffer.Get<float>(),proposal=proposal_buffer.Get<float>();
    auto summary=summary_buffer.Get<float>(),coefficient=coefficient_buffer.Get<float>();
    const I count=((__gm__ I*)lengths)[1],tiles=(width+tile-1)/tile,stride=5*width+2;
    // Metadata preflight establishes sorted owners and increasing int64 clocks.
    // Each owner/tile has one writer; its actual time sequence stays in order.
    for(I task=AscendC::GetBlockIdx();task<count*tiles;task+=AscendC::GetBlockNum()) {
      const I first=task/tiles,start=(task%tiles)*tile;
      const I b=events[first*13],n=events[first*13+1],key=b*nodes+n,kind=cfg[n*3];
      if(first&&events[(first-1)*13]==b&&events[(first-1)*13+1]==n)continue;
      const uint32_t size=width-start<tile?width-start:tile;
      load(old,s,key*width+start,size);
      if(kind==1)load(coefficient,a,n*width+start,size);
      for(I i=first;i<count&&events[i*13]==b&&events[i*13+1]==n;++i) {
        const bool active=events[i*13+3],adopt=cfg[n*3+2]||active,clear=cfg[n*3+1]&&active;
        if(kind||trace)load(summary,h,i*width+start,size);
        if(trace){save(summary,v,i*stride+start,size);save(old,v,i*stride+width+start,size);}
        AscendC::Muls(proposal,old,1.0f,size);AscendC::PipeBarrier<PIPE_V>();
        if(kind==1) {
          AscendC::Mul(proposal,old,coefficient,size);AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add(proposal,proposal,summary,size);AscendC::PipeBarrier<PIPE_V>();
        } else if(kind==2) {
          const float retention_value=rho[n];
          const auto local=tide_device::local_time_unchecked(events[i*13+2],(__gm__ I*)policy,n);
          const auto local_old=tide_device::local_time_unchecked(events[i*13+4],(__gm__ I*)policy,n);
          const uint64_t ticks=(uint64_t(local)+1)-uint64_t(local_old+1);
          for(uint64_t tick=0;tick<ticks;++tick) {
            AscendC::Muls(proposal,proposal,retention_value,size);AscendC::PipeBarrier<PIPE_V>();
          }
          AscendC::Add(proposal,summary,proposal,size);AscendC::PipeBarrier<PIPE_V>();
        }
        if(kind==3)load(proposal,ap,i*width+start,size);
        if(trace)save(proposal,v,i*stride+2*width+start,size);
        if(adopt){AscendC::Muls(old,proposal,1.0f,size);AscendC::PipeBarrier<PIPE_V>();}
        save(old,c,i*width+start,size);
        if(trace)save(old,v,i*stride+3*width+start,size);
        if(clear){AscendC::Muls(old,old,0.0f,size);AscendC::PipeBarrier<PIPE_V>();}
        if(trace)save(old,v,i*stride+4*width+start,size);
      }
      save(old,s,key*width+start,size);
    }
  }
 private:
  __aicore__ inline void load(AscendC::LocalTensor<float> to,AscendC::GlobalTensor<float>& from,I offset,uint32_t size) {
    auto x=input.AllocTensor<float>();
    const AscendC::DataCopyExtParams copy{1,size*uint32_t(sizeof(float)),0,0,0};
    const AscendC::DataCopyPadExtParams<float> padding{true,0,uint8_t((8-size%8)%8),0};
    AscendC::DataCopyPad(x,from[offset],copy,padding);
    input.EnQue(x);x=input.DeQue<float>();AscendC::Muls(to,x,1.0f,size);
    AscendC::PipeBarrier<PIPE_V>();input.FreeTensor(x);
  }
  __aicore__ inline void save(AscendC::LocalTensor<float> from,AscendC::GlobalTensor<float>& to,I offset,uint32_t size) {
    auto x=output.AllocTensor<float>();AscendC::Muls(x,from,1.0f,size);
    output.EnQue(x);x=output.DeQue<float>();
    const AscendC::DataCopyExtParams copy{1,size*uint32_t(sizeof(float)),0,0,0};
    AscendC::DataCopyPad(to[offset],x,copy);output.FreeTensor(x);
  }
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN,1> input;
  AscendC::TQue<AscendC::QuePosition::VECOUT,1> output;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> old_buffer,proposal_buffer,summary_buffer,coefficient_buffer;
};
}
extern "C" __global__ __aicore__ void tide_vector_state(GM_ADDR lengths,GM_ADDR content,
    GM_ADDR config,GM_ADDR coefficients,GM_ADDR retention,GM_ADDR policy,GM_ADDR state,GM_ADDR comparisons,
    GM_ADDR metadata,GM_ADDR diagnostics,GM_ADDR proposals,GM_ADDR error,int64_t width,int64_t nodes,int64_t trace) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  VectorState op;op.run(lengths,content,config,coefficients,retention,policy,state,comparisons,metadata,diagnostics,proposals,error,width,nodes,trace);
}
