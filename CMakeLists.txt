cmake_minimum_required(VERSION 3.0)
project(SIMD_Experiments)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules)
include_directories(include)

set(CORE_CXX_FLAGS "-pthread -Wstrict-aliasing -Wall -ffast-math -march=native -mavx -msse3")
if (NOT WIN32)
    set(EXTRA_DEBUG_FLAGS "-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer")
else()
    set(EXTRA_DEBUG_FLAGS "")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CORE_CXX_FLAGS} ${EXTRA_DEBUG_FLAGS} -g3 -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CORE_CXX_FLAGS} -O3")

add_subdirectory(number_guess)
