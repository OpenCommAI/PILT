cmake_minimum_required(VERSION 3.16)
project(libproto CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter
                    -Wno-unused-but-set-variable)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ── PILT (full source set: encoder + aggregator + wire) ─────────────────────
add_library(pilt STATIC
    src/wire.cpp
    src/encoder.cpp
    src/aggregator.cpp
)

# ── DCTCP, LTP, PLOT are header-only (logic + dense codec from common/) ────
add_library(proto_baselines INTERFACE)
target_include_directories(proto_baselines INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# ── Tests ──────────────────────────────────────────────────────────────────
enable_testing()
foreach(t test_wire test_encoder test_round_trip test_dctcp_ltp_plot)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE pilt)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

# ── PILT TCP demo binaries ─────────────────────────────────────────────────
add_executable(pilt_worker            demo/worker.cpp)
add_executable(pilt_parameter_server  demo/parameter_server.cpp)
target_link_libraries(pilt_worker            PRIVATE pilt pthread)
target_link_libraries(pilt_parameter_server  PRIVATE pilt pthread)

# ── 4-protocol head-to-head benchmark ──────────────────────────────────────
add_executable(compare_protocols bench/compare_protocols.cpp)
target_link_libraries(compare_protocols PRIVATE pilt)

# Convenient `make check`: build everything, run ctest.
add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_wire test_encoder test_round_trip test_dctcp_ltp_plot
)
