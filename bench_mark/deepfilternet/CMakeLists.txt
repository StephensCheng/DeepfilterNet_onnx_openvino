cmake_minimum_required(VERSION 3.12)
project(DeepFilterNet)
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")
# set(LIBTORCH ${CMAKE_SOURCE_DIR}/libtorch/)
# set(CMAKE_PREFIX_PATH ${LIBTORCH}/share/cmake/Torch)

message("Torch libraries: ${TORCH_LIBRARIES}")
find_package(Torch REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

set(LIBDEEPFILTER ${CMAKE_SOURCE_DIR}/model/onnx/lib/libdeepfilter.a
    openvino::runtime
)

# Add your source files here
set(SOURCES
    src/main.cpp
    src/musicgen_utils.cpp
    src/dfnet_model.cpp
    src/deepfilter.cpp
)

# Add your include directories here
include_directories(
    ${CMAKE_SOURCE_DIR}/model/onnx/include
    ${CMAKE_SOURCE_DIR}/musicgen_utils.hpp
    ${CMAKE_SOURCE_DIR}/src/dfnet_model.hpp
    ${CMAKE_SOURCE_DIR}/src/deepfilter.hpp
)

# Add your executable target here
add_executable(DeepFilterNet ${SOURCES})

# Link your libraries here
target_link_libraries(DeepFilterNet ${LIBDEEPFILTER} openvino::runtime ${TORCH_LIBRARIES})
