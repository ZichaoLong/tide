cmake_minimum_required(VERSION 3.18)
project(tidegraph LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(tidegraph-lh-scope-check cpp/test/lh_scope_check.cpp)
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)
add_executable(tidegraph-profile-check cpp/test/operator_profile.cpp)
target_include_directories(tidegraph-profile-check PRIVATE cpp/include)
target_link_libraries(tidegraph-profile-check PRIVATE Threads::Threads)
add_library(tidegraph STATIC
  cpp/src/graph.cpp cpp/src/ports.cpp cpp/src/ops.cpp cpp/src/full_kernel.cpp cpp/src/full_evaluate.cpp cpp/src/isolated_linear.cpp cpp/src/full_rows.cpp cpp/src/row_emit.cpp cpp/src/pool.cpp cpp/src/dense.cpp
  cpp/src/aggregate_kernel.cpp cpp/src/aggregate_evaluate.cpp cpp/src/isolated_aggregate.cpp cpp/src/source_domain.cpp cpp/src/read.cpp cpp/src/next.cpp
  cpp/src/region.cpp cpp/src/region_evaluate.cpp cpp/src/lh_selector.cpp cpp/src/lazy_add.cpp cpp/src/lh_full.cpp
  cpp/src/fiber_attention.cpp cpp/src/fiber_packing.cpp cpp/src/fiber_single.cpp cpp/src/fiber_rows.cpp
  cpp/src/fiber_pool.cpp cpp/src/fiber_pool_csr.cpp cpp/src/token_window.cpp
  cpp/src/validation.cpp cpp/src/input_validation.cpp cpp/src/stream.cpp cpp/src/stream_support.cpp cpp/src/cursor.cpp cpp/src/planner.cpp
  cpp/src/block.cpp cpp/src/block_step.cpp cpp/src/frontier.cpp cpp/src/specialized.cpp cpp/src/settle.cpp cpp/src/settle_projection.cpp
  cpp/src/kernel.cpp cpp/src/clock.cpp cpp/src/clocked_kernel.cpp cpp/src/autograd.cpp cpp/src/basic_kernels.cpp cpp/src/matrix_kernels.cpp
  cpp/src/packed.cpp cpp/src/attention.cpp cpp/src/block_prepare.cpp cpp/src/parameters.cpp cpp/src/optimizer.cpp
  cpp/src/checkpoint.cpp cpp/src/checkpoint_codec.cpp cpp/src/checkpoint_preflight.cpp cpp/src/checkpoint_io.cpp)
set_target_properties(tidegraph PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tidegraph PUBLIC cpp/include)
target_link_libraries(tidegraph PUBLIC torch Threads::Threads)
separate_arguments(TIDE_TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(tidegraph PUBLIC ${TIDE_TORCH_FLAGS})
add_executable(tidegraph-smoke cpp/src/main.cpp cpp/src/runtime.cpp)
target_link_libraries(tidegraph-smoke PRIVATE tidegraph)
add_executable(tidegraph-settle-check cpp/test/settle.cpp cpp/src/runtime.cpp)
target_link_libraries(tidegraph-settle-check PRIVATE tidegraph)
add_executable(tidegraph-kernel-check cpp/test/custom_kernel.cpp cpp/test/content_programs.cpp cpp/test/read_programs.cpp cpp/test/next_programs.cpp cpp/test/region_programs.cpp cpp/test/fiber_policy.cpp cpp/test/state_clocks.cpp cpp/src/runtime.cpp)
target_link_libraries(tidegraph-kernel-check PRIVATE tidegraph)
add_executable(tidegraph-full-check cpp/test/custom_full.cpp cpp/src/runtime.cpp)
target_link_libraries(tidegraph-full-check PRIVATE tidegraph)
add_executable(tidegraph-aggregate-check cpp/test/custom_aggregate.cpp cpp/src/runtime.cpp)
target_link_libraries(tidegraph-aggregate-check PRIVATE tidegraph)
add_executable(tidegraph-optimizer-check cpp/test/optimizer.cpp cpp/src/runtime.cpp)
target_link_libraries(tidegraph-optimizer-check PRIVATE tidegraph)
add_executable(tidegraph-checkpoint-check cpp/test/checkpoint.cpp cpp/src/runtime.cpp)
target_link_libraries(tidegraph-checkpoint-check PRIVATE tidegraph)
add_executable(tidegraph-streaming-bench cpp/bench/main.cpp cpp/bench/config.cpp
  cpp/bench/workload.cpp cpp/bench/compare.cpp cpp/src/runtime.cpp)
target_link_libraries(tidegraph-streaming-bench PRIVATE tidegraph)
add_executable(tidegraph-scale-bench cpp/scale/main.cpp cpp/scale/config.cpp
  cpp/scale/model.cpp cpp/scale/check.cpp cpp/scale/grad_check.cpp cpp/bench/compare.cpp cpp/src/runtime.cpp cpp/src/threads.cpp)
target_link_libraries(tidegraph-scale-bench PRIVATE tidegraph ${CMAKE_DL_LIBS})
option(TIDE_PYTHON_BINDINGS "Build the Python test/client adapter" ON)
if(TIDE_PYTHON_BINDINGS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED NO_DEFAULT_PATH)
  Python3_add_library(_tide_native MODULE cpp/src/bindings.cpp cpp/src/bind_settle.cpp cpp/src/bind_metrics.cpp cpp/src/bind_full.cpp cpp/src/bind_aggregate.cpp)
  target_link_libraries(_tide_native PRIVATE tidegraph ${TORCH_PYTHON_LIBRARY})
endif()
