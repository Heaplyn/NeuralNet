cmake_minimum_required(VERSION 3.20)
project(RingWrapperNeuralNet VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(NOT MSVC)
    find_package(OpenMP)
endif()

# Source files grouped by Ring Architecture
set(RING0_SOURCES
    src/ring0/tensor.cpp
    src/ring0/activations.cpp
    src/ring0/loss.cpp
    src/ring0/taylor_predictor.cpp
    src/ring0/cuda_backend.cpp
    src/ring0/calculus_of_constructions.cpp
)

set(RING1_SOURCES
    src/ring1/layer.cpp
    src/ring1/optimizer.cpp
    src/ring1/embedding.cpp
    src/ring1/attention.cpp
    src/ring1/transformer_block.cpp
    src/ring1/adamw.cpp
    src/ring1/recursive_layer.cpp
    src/ring1/meta_loss_optimizer.cpp
    src/ring1/multi_formula_optimizer.cpp
    src/ring1/dependent_type_attention.cpp
)

set(RING2_SOURCES
    src/ring2/neural_net.cpp
    src/ring2/growth_controller.cpp
    src/ring2/tokenizer.cpp
    src/ring2/transformer_lm.cpp
    src/ring2/vocab_manager.cpp
)

set(RING3_SOURCES
    src/ring3/letter_dataset.cpp
    src/ring3/mnist_dataset.cpp
    src/ring3/image_dataset.cpp
    src/ring3/text_dataset.cpp
    src/ring3/data_loader.cpp
    src/ring3/chrono_scheduler.cpp
    src/ring3/llm_trainer.cpp
    src/ring3/trainer.cpp
)

# Core RingWrapper Library
add_library(ringwrapper_nn STATIC
    ${RING0_SOURCES}
    ${RING1_SOURCES}
    ${RING2_SOURCES}
    ${RING3_SOURCES}
)

target_include_directories(ringwrapper_nn PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ringwrapper_nn PUBLIC OpenMP::OpenMP_CXX)
endif()

# Ring 4 Demo Application
add_executable(nn_demo
    src/main.cpp
)

target_link_libraries(nn_demo PRIVATE ringwrapper_nn)

if(OpenMP_CXX_FOUND)
    target_link_libraries(nn_demo PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(ringwrapper_nn PRIVATE /W4 $<$<NOT:$<CONFIG:Debug>>:/O2> /fp:fast /openmp)
    target_compile_options(nn_demo PRIVATE /W4 $<$<NOT:$<CONFIG:Debug>>:/O2> /fp:fast /openmp)
else()
    target_compile_options(ringwrapper_nn PRIVATE -Wall -Wextra -O3 -march=native -ffast-math -fopenmp)
    target_compile_options(nn_demo PRIVATE -Wall -Wextra -O3 -march=native -ffast-math -fopenmp)
endif()
