#pragma once
#include "tide/types.h"

namespace tide::device_online {
// Static node ownership only. This plan contains no input-dependent events,
// selections or routes. Device indices are logical indices supplied by a client.
struct FullPlacement {
  std::vector<at::Device> devices;
  std::vector<int64_t> owners;
};
FullPlacement place_full(const Graph&,const Model&,std::vector<at::Device>,const std::string& policy);
void validate_full_placement(const FullPlacement&,int64_t nodes,at::Device coordinator);
} // namespace tide::device_online
