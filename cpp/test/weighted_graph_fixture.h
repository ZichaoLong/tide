#pragma once
#include "tide/weighted.h"
namespace weighted_check {
tide::Graph graph(const std::string& topology,const std::string& family);
tide::Model model(const tide::Graph&,const std::string& family,at::Device,at::ScalarType);
std::vector<tide::External> inputs(const tide::Graph&,at::Device,at::ScalarType,tide::Index start=0);
}
