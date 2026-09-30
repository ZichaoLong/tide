#include "kernel_operator.h"
namespace {
using I=int64_t;
constexpr uint32_t tile=256;
// A single writer owns each (sample,node,width tile). Its local state follows
// that owner's actual time order; independent owners/width tiles run in parallel.
class VectorRead {
 public:
  __aicore__ inline void run(GM_ADDR fibers,GM_ADDR lengths,GM_ADDR content,GM_ADDR reads,
      GM_ADDR modes,GM_ADDR kinds,GM_ADDR config,GM_ADDR coefficients,GM_ADDR retention,
      GM_ADDR initial,GM_ADDR ticks,GM_ADDR partials,GM_ADDR proposals,GM_ADDR error,I width,I nodes) {
    AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)fibers);
    AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
    if(((__gm__ int32_t*)error)[0])return;
    auto f=(__gm__ I*)fibers,m=(__gm__ I*)modes,k=(__gm__ I*)kinds,cfg=(__gm__ I*)config;
    auto steps=(__gm__ I*)ticks;auto rho=(__gm__ float*)retention;
    AscendC::GlobalTensor<float> h,w,a,s,p,ap;
    h.SetGlobalBuffer((__gm__ float*)content);w.SetGlobalBuffer((__gm__ float*)reads);
    a.SetGlobalBuffer((__gm__ float*)coefficients);s.SetGlobalBuffer((__gm__ float*)initial);
    p.SetGlobalBuffer((__gm__ float*)partials);ap.SetGlobalBuffer((__gm__ float*)proposals);
    pipe.InitBuffer(input,1,tile*sizeof(float));pipe.InitBuffer(output,1,32);
    pipe.InitBuffer(old_buffer,tile*sizeof(float));pipe.InitBuffer(proposal_buffer,tile*sizeof(float));
    pipe.InitBuffer(summary_buffer,tile*sizeof(float));pipe.InitBuffer(coefficient_buffer,tile*sizeof(float));
    pipe.InitBuffer(weight_buffer,tile*sizeof(float));pipe.InitBuffer(product_buffer,tile*sizeof(float));
    pipe.InitBuffer(reduction_buffer,tile*sizeof(float));pipe.InitBuffer(sum_buffer,32);
    auto old=old_buffer.Get<float>(),proposal=proposal_buffer.Get<float>(),summary=summary_buffer.Get<float>();
    auto coefficient=coefficient_buffer.Get<float>(),weight=weight_buffer.Get<float>();
    auto product=product_buffer.Get<float>(),temporary=reduction_buffer.Get<float>(),sum=sum_buffer.Get<float>();
    const I count=((__gm__ I*)lengths)[1],tiles=(width+tile-1)/tile;
    for(I task=AscendC::GetBlockIdx();task<count*tiles;task+=AscendC::GetBlockNum()) {
      const I first=task/tiles,part=task%tiles,start=part*tile,b=f[first*4],n=f[first*4+1];
      if(first&&f[(first-1)*4]==b&&f[(first-1)*4+1]==n)continue;
      const I mode=m[n],kind=cfg[n*3];const bool norm=k[n];
      const uint32_t size=width-start<tile?width-start:tile;
      if(mode>0){load(old,s,(b*nodes+n)*width+start,size);if(kind==1)load(coefficient,a,n*width+start,size);}
      if(mode>=0&&!norm)load(weight,w,n*width+start,size);
      for(I row=first;row<count&&f[row*4]==b&&f[row*4+1]==n;++row) {
        if(mode<0){AscendC::Duplicate(sum,0.f,8);AscendC::PipeBarrier<PIPE_V>();save(sum,p,row*tiles+part);continue;}
        load(summary,h,row*width+start,size);
        auto visible=summary;
        if(mode>0) {
          AscendC::Muls(proposal,old,1.f,size);AscendC::PipeBarrier<PIPE_V>();
          if(kind==1) {
            AscendC::Mul(proposal,old,coefficient,size);AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(proposal,proposal,summary,size);AscendC::PipeBarrier<PIPE_V>();
          }else if(kind==2) {
            const float retention_value=rho[n];
            for(I tick=0;tick<steps[row];++tick){AscendC::Muls(proposal,proposal,retention_value,size);AscendC::PipeBarrier<PIPE_V>();}
            AscendC::Add(proposal,summary,proposal,size);AscendC::PipeBarrier<PIPE_V>();
          }
          if(kind==3)load(proposal,ap,row*width+start,size);
          visible=mode==1?old:proposal;
        }
        AscendC::Mul(product,visible,norm?visible:weight,size);AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(sum,product,temporary,int32_t(size));AscendC::PipeBarrier<PIPE_V>();
        save(sum,p,row*tiles+part);
        if(mode>0){AscendC::Muls(old,proposal,1.f,size);AscendC::PipeBarrier<PIPE_V>();}
      }
    }
  }
 private:
  __aicore__ inline void load(AscendC::LocalTensor<float> to,AscendC::GlobalTensor<float>& from,I offset,uint32_t size) {
    auto x=input.AllocTensor<float>();
    const AscendC::DataCopyExtParams copy{1,size*uint32_t(sizeof(float)),0,0,0};
    const AscendC::DataCopyPadExtParams<float> padding{true,0,uint8_t((8-size%8)%8),0};
    AscendC::DataCopyPad(x,from[offset],copy,padding);input.EnQue(x);x=input.DeQue<float>();
    AscendC::Muls(to,x,1.f,size);AscendC::PipeBarrier<PIPE_V>();input.FreeTensor(x);
  }
  __aicore__ inline void save(AscendC::LocalTensor<float> from,AscendC::GlobalTensor<float>& to,I offset) {
    auto x=output.AllocTensor<float>();AscendC::Muls(x,from,1.f,1);output.EnQue(x);x=output.DeQue<float>();
    const AscendC::DataCopyExtParams copy{1,sizeof(float),0,0,0};
    AscendC::DataCopyPad(to[offset],x,copy);output.FreeTensor(x);
  }
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN,1> input;
  AscendC::TQue<AscendC::QuePosition::VECOUT,1> output;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> old_buffer,proposal_buffer,summary_buffer,coefficient_buffer;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> weight_buffer,product_buffer,reduction_buffer,sum_buffer;
};
}
extern "C" __global__ __aicore__ void tide_vector_read(GM_ADDR fibers,GM_ADDR lengths,
    GM_ADDR content,GM_ADDR reads,GM_ADDR modes,GM_ADDR kinds,GM_ADDR config,GM_ADDR coefficients,
    GM_ADDR retention,GM_ADDR initial,GM_ADDR ticks,GM_ADDR partials,GM_ADDR proposals,GM_ADDR error,int64_t width,int64_t nodes) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  VectorRead op;op.run(fibers,lengths,content,reads,modes,kinds,config,coefficients,retention,initial,ticks,partials,proposals,error,width,nodes);
}
