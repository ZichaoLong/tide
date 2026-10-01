#pragma once
#include "content_fixture.h"
#include "graph_vjp.h"
#include <map>

namespace tide::device_online::test {
Fixture graph_vjp_fixture(int shape,int variant,int64_t width=3);
struct GraphReference {
  Result result;
  std::map<std::string,at::Tensor> gradients;
};
GraphReference graph_reference(Fixture,int mode,bool warm,at::ScalarType);
GraphReference graph_reference_precision(Fixture,int mode,bool warm,at::ScalarType,bool half,Options={});
std::string boundary_name(const Atom&);
void compare_graph_vjp(const ReverseTape&,const GraphVjp&,const GraphReference&);
void compare_graph_vjp_precision(const ReverseTape&,const GraphVjp&,const GraphReference&,bool half);
} // namespace tide::device_online::test
