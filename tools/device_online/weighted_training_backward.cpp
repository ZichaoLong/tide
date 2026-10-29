#include "weighted_training_internal.h"
#include <chrono>
namespace tide::device_online {
namespace {
ResidentParameterGradient public_parameters(const ParameterVjp& x){
  ResidentParameterGradient out;out.offsets=x.offsets;out.values=x.values;out.connected=x.connected;
  for(const auto& p:x.owners){out.names.push_back(p.canonical);out.aliases.push_back(p.aliases);}return out;
}
}
WeightedGradients WeightedTrainingSession::Impl::reverse(const std::vector<WeightedCotangents>& roots){
  check();
  if(gradients_ready||saved.empty()||roots.size()!=saved.size())throw std::logic_error("weighted backward requires all unconsumed windows in order");
  int64_t largest_view=0;long double output_bytes=0;
  for(size_t i=0;i<saved.size();++i){validate_weighted_cotangents(saved[i].view,roots[i]);
    largest_view=std::max(largest_view,weighted_training_view_bytes(saved[i].view));
    output_bytes+=2.L*saved[i].snapshot->limits.queue*(50.L+4.L*saved[i].snapshot->model.width());}
  for(const auto& l:saved.front().snapshot->layouts)output_bytes+=saved.front().snapshot->samples*(4.L*l.floating_width()+8.L*l.integer_width()+l.fields().size()+1);
  // Output references, root packing temporaries, complete parameter partials and
  // the bridge coexist with at most two adjoints. All fit one backward quota.
  const long double auxiliary=output_bytes+4.L*largest_view+4.L*(layout.values.nbytes()+layout.connected.nbytes())+4096;
  if(auxiliary>=limits.backward_bytes)throw std::invalid_argument("weighted backward root/output reservation exceeds budget");
  const int64_t envelope=limits.backward_bytes-static_cast<int64_t>(auxiliary);
  WeightedGradients result;result.boundaries.resize(saved.size());ParameterVjp total;
  std::unique_ptr<WeightedWindowAdjoint> later;int64_t peak=0;
  using Clock=std::chrono::steady_clock;
  auto micros=[](Clock::time_point start){return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-start).count();};
  int64_t construct_us=0,seed_us=0,run_us=0,bridge_us=0,reduce_us=0;
  auto error=at::zeros({1},layout.values.options().dtype(at::kInt));
  try{
    for(size_t index=saved.size();index>0;){--index;
      const auto later_bytes=later?later->reserved_bytes():0;
      if(later_bytes>=envelope)throw std::invalid_argument("weighted adjacent adjoints exceed backward budget");
      auto started=Clock::now();
      auto current=std::make_unique<WeightedWindowAdjoint>(saved[index].snapshot,envelope-later_bytes);
      construct_us+=micros(started);
      const auto used=current->reserved_bytes()+later_bytes;peak=std::max(peak,used);
      started=Clock::now();seed_weighted_cotangents(*saved[index].snapshot,saved[index].view,roots[index],current->roots());seed_us+=micros(started);
      if(later){
        started=Clock::now();
        DeviceProgram bridge(device);bridge.limit_workspace(limits.forward.operator_bytes);
        append_weighted_window_bridge(bridge,*saved[index].snapshot,current->roots(),*saved[index+1].snapshot,*later,error,envelope-used);
        if(bridge.retained_tensor_bytes()+used>limits.backward_bytes)throw std::invalid_argument("weighted bridge exceeds backward reservation");
        run(bridge,error);later->close();later.reset();bridge_us+=micros(started);
      }
      started=Clock::now();current->run();run_us+=micros(started);
      started=Clock::now();
      DeviceProgram reduce(device);reduce.limit_workspace(limits.forward.operator_bytes);
      auto part=append_weighted_training_parameters(reduce,current->parameters(),layout,envelope-current->reserved_bytes());
      auto combined=total.values.defined()?append_parameter_accumulate(reduce,total,part,error,envelope-current->reserved_bytes()):part;
      if(reduce.retained_tensor_bytes()+current->reserved_bytes()+output_bytes>limits.backward_bytes)
        throw std::invalid_argument("weighted parameter reduction exceeds backward reservation");
      run(reduce,error);total=std::move(combined);reduce_us+=micros(started);
      const auto& b=current->boundary();result.boundaries[index]={saved[index].view.token,b.coordinates,b.values,b.present,b.connected};
      if(index==0){const auto& snapshot=*saved[index].snapshot;
        for(size_t v=0;v<snapshot.layouts.size();++v){const auto& l=snapshot.layouts[v];const auto& g=current->states()[v];
          const auto prefix="node."+std::to_string(v)+".";WeightedStateGradient state;
          state.node=v;state.family=l.family();state.integer=snapshot.journal->boundary(prefix+"integer");state.present=snapshot.journal->boundary(prefix+"present").ne(0);
          for(size_t j=0;j<l.fields().size();++j){const auto& f=l.fields()[j];std::vector<int64_t> shape{snapshot.samples};
            if(f.ragged)shape.push_back(l.capacity());shape.insert(shape.end(),f.shape.begin(),f.shape.end());
            state.fields[f.name]={g.value.narrow(1,f.offset,f.width*(f.ragged?l.capacity():1)).reshape(shape),g.connected.select(1,j)};}
          result.initial.push_back(std::move(state));}
      }
      later=std::move(current);
    }
    later->close();later.reset();
    gradient=std::move(total);gradients_ready=true;
    guard_weighted_training(gradient_versions,gradient.values);guard_weighted_training(gradient_versions,gradient.connected);
    result.parameters=public_parameters(gradient);
    result.statistics={{"windows",static_cast<int64_t>(saved.size())},{"retained_reservation",saved_bytes},
      {"adjoint_peak_tensor_reservation",peak},{"backward_auxiliary_reservation",static_cast<int64_t>(auxiliary)},
      {"adjoint_construct_us",construct_us},{"root_seed_us",seed_us},{"adjoint_run_us",run_us},
      {"window_bridge_us",bridge_us},{"parameter_reduce_us",reduce_us}};
    saved.clear();saved_bytes=0;return result;
  }catch(const std::invalid_argument&){throw;}catch(...){failed=true;throw;}
}
WeightedGradients WeightedTrainingSession::backward(const std::vector<WeightedCotangents>& roots){return impl_->reverse(roots);}
}
