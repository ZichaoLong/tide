#pragma once
#include "checkpoint_internal.h"
namespace tide::checkpoint_detail {
struct GraphRecord { Decoded parameters; Continuation continuation; };
std::vector<uint8_t> encode_graph(const GraphRecord&);
GraphRecord decode_graph(const std::vector<uint8_t>&);
}
