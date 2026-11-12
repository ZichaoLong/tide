// One compact private graph, one batched first-order transpose, exact root support.
#include "weighted_kv_tensor.h"
#include "weighted_column_blocks.h"
#include "tide/autograd_census.h"
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/variable.h>
#include <unordered_map>
#include <unordered_set>

namespace tide::weighted::columns::tensor {
namespace {
using torch::autograd::variable_list;
using Node=torch::autograd::Node;
std::vector<std::vector<bool>> dependencies(const variable_list& fields,const std::vector<Role>& roles){
  std::unordered_map<Node*,size_t> leaves;
  for(size_t r=0;r<roles.size();++r)leaves[torch::autograd::impl::gradient_edge(roles[r].value).function.get()]=r;
  std::vector<std::vector<bool>> result(fields.size(),std::vector<bool>(roles.size(),false));
  for(size_t f=0;f<fields.size();++f){if(!fields[f].requires_grad())continue;
    std::vector<std::shared_ptr<Node>> pending{torch::autograd::impl::gradient_edge(fields[f]).function};
    std::unordered_set<Node*> visited;
    while(!pending.empty()){auto node=pending.back();pending.pop_back();if(!node||!visited.insert(node.get()).second)continue;
      auto leaf=leaves.find(node.get());if(leaf!=leaves.end()){result[f][leaf->second]=true;continue;}
      for(const auto& edge:node->next_edges())if(edge.function)pending.push_back(edge.function);
    }
  }
  return result;
}
class Boundary final:public Node {
 public:
  Packed packed;
  variable_list fields;
  std::vector<std::vector<bool>> uses;
  std::vector<std::pair<size_t,size_t>> outputs; // node, field; only differentiable publications.
  std::vector<torch::autograd::SavedVariable> originals;
  variable_list apply(variable_list&& bars)override{
    if(at::GradMode::is_enabled())throw std::invalid_argument("KV tensor boundary supports first-order VJP only");
    if(originals.empty())throw std::runtime_error("KV tensor boundary graph released");
    for(auto& value:originals)value.unpack(); // Guard original storage versions, including aliases.
    const auto nodes=packed.nodes;variable_list result(originals.size());
    std::vector<std::vector<bool>> connected(packed.roles.size(),std::vector<bool>(nodes,false));
    variable_list active_fields,active_bars;
    for(size_t f=0;f<fields.size();++f){variable_list pieces,indices;
      for(size_t j=0;j<bars.size();++j)if(outputs[j].second==f&&bars[j].defined()){
        const auto g=outputs[j].first;pieces.push_back(bars[j]);indices.push_back(packed.publish[g]);
        for(size_t r=0;r<packed.roles.size();++r)if(uses[f][r])connected[r][g]=true;
      }
      if(pieces.empty())continue;
      // Fields may be slices of one KV projection. A contiguous adjoint makes
      // flatten a view; preserving an arbitrary field stride could copy here.
      auto bar=at::zeros(fields[f].sizes(),fields[f].options());bar.flatten(0,2).index_copy_(0,at::cat(indices),blocks::concatenate(pieces));
      active_fields.push_back(fields[f]);active_bars.push_back(bar);
    }
    if(active_fields.empty())return result;
    variable_list leaves;for(const auto& role:packed.roles)leaves.push_back(role.value);
    // Retain the private graph until release_variables: public independent roots
    // may invoke this node repeatedly with retain_graph=true.
    auto gradients=torch::autograd::grad(active_fields,leaves,active_bars,true,false,true);
    for(size_t r=0;r<packed.roles.size();++r){const auto& role=packed.roles[r];if(!gradients[r].defined())continue;
      bool needed=false;for(Index g=0;g<nodes;++g)needed|=connected[r][g]&&should_compute_output(r*nodes+g);
      if(!needed)continue;
      if(role.shared){
        // A shared leaf already sums every used node. Publish its total on one
        // of the identical original edges, rather than multiplying it by G.
        for(Index g=0;g<nodes;++g)if(connected[r][g]&&should_compute_output(r*nodes+g)){result[r*nodes+g]=gradients[r];break;}
        continue;
      }
      Tensor restored;if(!role.parameter)restored=gather(gradients[r].flatten(0,2),role.restore);
      Index offset=0;
      for(Index g=0;g<nodes;++g){const size_t slot=r*nodes+g;
        if(connected[r][g]&&should_compute_output(slot))result[slot]=role.parameter?gradients[r].select(0,g):restored.narrow(0,offset,role.lengths[g]);
        if(!role.parameter)offset+=role.lengths[g];
      }
    }
    return result;
  }
  void release_variables()override{originals.clear();fields.clear();packed=Packed{};uses.clear();outputs.clear();}
  std::string name()const override{return "tide::weighted::KVTensorBoundaryVjp";}
};
}
std::vector<TensorFields> evaluate(Packed packed,const std::function<TensorFields(const Packed&)>& compute,const char* label){
  const bool enabled=at::GradMode::is_enabled();at::NoGradGuard guard;
  bool grad=false;variable_list originals;
  for(auto& role:packed.roles)for(const auto& t:role.originals){originals.push_back(t);grad|=enabled&&t.requires_grad();}
  if(grad)for(auto& role:packed.roles)role.value.set_requires_grad(true);
  TensorFields values;{at::AutoGradMode graph(grad);values=compute(packed);}
  variable_list fields;std::vector<std::string> names;
  for(const auto& [name,value]:values){names.push_back(name);fields.push_back(value);}
  std::shared_ptr<Boundary> node;
  if(grad){node=std::make_shared<Boundary>();node->fields=fields;node->uses=dependencies(fields,packed.roles);
    node->set_next_edges(torch::autograd::collect_next_edges(originals));
    for(const auto& value:originals)node->originals.emplace_back(value,false);
    if(autograd_census::enabled()){
      autograd_census::graph(label,fields);
      autograd_census::saved(originals);autograd_census::saved(fields);
      for(const auto& role:packed.roles)autograd_census::saved({role.value,role.restore});
      autograd_census::saved({packed.age,packed.visible,packed.query_position,packed.key_position});
      autograd_census::saved(packed.publish);
    }
  }
  std::vector<TensorFields> out(packed.nodes);
  std::vector<Index> lengths;for(const auto& index:packed.publish)lengths.push_back(index.numel());
  const auto publish=at::cat(packed.publish);
  for(size_t f=0;f<fields.size();++f){
    auto columns=blocks::FieldBlock::publish(fields[f],publish,lengths).columns();
    for(size_t g=0;g<out.size();++g){auto value=columns[g];bool used=false;
      if(grad)for(size_t r=0;r<packed.roles.size();++r)used|=node->uses[f][r]&&packed.roles[r].originals[g].requires_grad();
      if(used){node->outputs.emplace_back(g,f);torch::autograd::set_history(value,node);}
      out[g][names[f]]=std::move(value);
    }
  }
  if(node){for(auto& role:packed.roles)role.originals.clear();node->packed=std::move(packed);}
  return out;
}
std::vector<TensorFields> evaluate(const std::vector<Request>& requests){
  return evaluate(pack(requests,false),[&](const Packed& p){return formula(requests[0],p);},
    requests[0].full?"private/kv.tensor.full":"private/kv.tensor.read");
}
}
