#pragma once
#include "tide/weighted.h"
namespace tide::device_online::weighted_flow_check {
std::pair<Graph,Model> configured(const weighted::RegionalConfig&,bool query,bool sources,bool origin);
std::vector<weighted::RegionalConfig> configurations();
}
