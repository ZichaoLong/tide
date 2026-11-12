#pragma once
// Fixed 0770593 KV §3–4 / allocation §3: one role tensor per node×B×Q/K group.
// Layout/timestamps remain int64; the numerical graph does not own scheduling.
#include "weighted_column_layout.h"
#include <set>

namespace tide::weighted::columns::tensor {
struct Coordinates {
  std::vector<Index> query,key;
  std::string reason="plain";
};
struct Request {
  const NodeWeights* owner;
  KVConfig config;
  RegionalConfig regional;
  const KVWindow* window;
  const Layout* bank_layout;
  const TensorFields* bank;
  Layout query_layout;
  std::vector<Index> rows;
  bool full=false, before=false, active=false;
  size_t output=0;
  Coordinates coordinates;
};
struct Role {
  std::string name;
  bool parameter=false,shared=false;
  std::vector<Tensor> originals;
  std::vector<Index> lengths;
  Tensor restore, value;
};
struct Packed {
  Index nodes=0,batch=0,queries=0,keys=0;
  std::vector<Role> roles;
  TensorFields values;
  std::set<std::string> shared;
  Tensor age,visible,query_position,key_position;
  std::vector<Tensor> publish;
};
bool eligible(const KVConfig&,const Tensor&);
Coordinates positions(const Request&);
std::vector<std::string> parameters(const Request&);
Packed pack(const std::vector<Request>&,bool differentiable);
TensorFields formula(const Request&,const Packed&);
std::vector<TensorFields> evaluate(const std::vector<Request>&);
void read_groups(const std::vector<KVReadRequest>&,const std::vector<size_t>&,Index,std::vector<TensorFields>&);
void full_groups(const std::vector<KVFullRequest>&,const std::vector<size_t>&,Index,std::vector<Message>&);
}
