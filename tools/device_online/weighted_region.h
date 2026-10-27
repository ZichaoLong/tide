#pragma once
// 0770593 allocation §2–4/S.1; docs/resident-weighted-region.md.
#include "weighted_head.h"
#include "weighted_allocation.h"
#include "weighted_sending.h"
#include "tide/weighted.h"
#include <optional>
namespace tide::device_online {
struct WeightedRegionFields {at::Tensor received,caps,score,evidence,current,key;};
struct WeightedRegionParameters {
  WeightedHeadParameters context,query;
  at::Tensor context_owners,query_owners;
};
struct WeightedRegionTape {
  weighted::RegionalConfig config;
  WeightedRegionFields fields;
  at::Tensor lengths,ids,times,context_input,query_input,statistics;
  at::Tensor context_rows,query_rows,context_values,query_values,scores,bounds,parameters;
  std::optional<WeightedHeadTape> context_head,query_head;
  std::optional<WeightedAllocationTape> allocation;
  std::optional<WeightedSendingTape> sending;
  at::Tensor controls,active,history_times,history_present;
};
struct WeightedRegionGradient {
  WeightedRegionFields fields;
  at::Tensor connected; // bool [rows,6], in WeightedRegionFields order.
  std::optional<WeightedHeadGradient> context_head,query_head;
};
// Whole candidate rows with private times/int64 IDs and explicit absence.
// Configuration is uniform within this numerical batch; graph owners may
// partition heterogeneous rows before appending independent batch programs.
WeightedRegionTape append_weighted_region(DeviceProgram&,const WeightedRegionFields&,
  const at::Tensor& ids,const at::Tensor& times,const at::Tensor& lengths,
  const WeightedRegionParameters&,const weighted::RegionalConfig&,const at::Tensor& error,
  int64_t scratch_budget=256*1024*1024);
WeightedRegionGradient append_weighted_region_vjp(DeviceProgram&,const WeightedRegionTape&,
  const at::Tensor& cotangent,const at::Tensor& roots,const at::Tensor& error,
  int64_t scratch_budget=256*1024*1024);
}
