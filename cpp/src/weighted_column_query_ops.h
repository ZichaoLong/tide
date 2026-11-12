#pragma once
// Independent whole-node roots in node×B contractions; caller owns admission.
#include "tide/weighted_sequence_columns.h"
namespace tide::weighted::columns {
std::vector<Tensor> project_nodes(const std::vector<Tensor>&,const std::vector<Tensor>&);
enum class Contraction {matmul,attention,logits};
std::vector<Tensor> contract_nodes(Contraction,const std::vector<std::vector<Tensor>>&);
}
