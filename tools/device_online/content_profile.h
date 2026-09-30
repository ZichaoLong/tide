#pragma once
#include "content_flow.h"
#include "frame_selector.h"
#include "broadcast_router.h"

namespace tide::device_online {
struct ContentProfile {
  Graph graph;
  Model model; // validated immutable CPU parameter snapshot
  int64_t width;
  std::vector<int64_t> owners;
  std::vector<Wire> wires;
  std::vector<SelectionPolicy> policies;
  std::vector<int64_t> causal_regions;
  bool all_content=true;
  // source rows: [target, logical slot]; source scales ordered input then edge.
  at::Tensor sources,origins,scales,read,read_modes,read_kinds,decay,retention,clock_policy,config;
  ContentProfile(Graph,Model,at::Device);
};
struct ContentState {at::Tensor values,clocks,present;};
struct ContentBatch {
  at::Tensor content,scores,weighted;
};
struct ContentUpdate {
  ContentState state;
  ActionBatch actions;
  at::Tensor comparison;
  at::Tensor event_meta,event_values;
};
ContentBatch append_content(CannProgram&,const ContentProfile&,const ReadyBatch&,const at::Tensor& error,bool vectorized);
void append_read(CannProgram&,const ContentProfile&,const ReadyBatch&,const ContentBatch&,
                 const ContentState&,const at::Tensor& coefficients,const at::Tensor& error,int64_t max_repeat_ticks,bool vectorized,
                 const at::Tensor& attention_proposals={});
ContentUpdate append_content_state(CannProgram&,const ContentProfile&,const ReadyBatch&,
    const ContentBatch&,const SelectionProposal&,const ContentState&,const at::Tensor& coefficients,
    const at::Tensor& stages,const at::Tensor& event_count,const at::Tensor& error,const ContentLimits&,
    const at::Tensor& attention_proposals={});
void commit_content_state(CannProgram&,const ContentState&,const ContentUpdate&,const at::Tensor& error);
} // namespace tide::device_online
