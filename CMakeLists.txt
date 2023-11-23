cmake_minimum_required(VERSION 3.5)


project(DistGER)

set(CMAKE_BUILD_TYPE "Release")

set(DGER_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
option(WITH_TESTS "Build unit test programs" OFF)
# 设置导出 compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# 设置 Debug 模式下的编译选项
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

# 设置 Release 模式下的编译选项
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

include(distger.cmake)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(MPI REQUIRED)
if(MPI_FOUND)
    set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
    set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
    set(MPI_EXECUTABLE_SUFFIX .openmpi)
    include_directories(${MPI_INCLUDE_PATH})
endif()

if (NOT TARGET spdlog)
    find_package(spdlog REQUIRED)
endif()

if(WITH_TESTS)
    message("Unit ON")
    enable_testing()
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/ext/gtest/googletest/include")
    add_subdirectory(ext/gtest)
    link_directories(${CMAKE_BINARY_DIR}/lib)
    set(GTEST_LIBRARIES "gtest" "gtest_main")
endif()

#use c++11

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++0x -D USE_MKL")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++0x ")
message("${CMAKE_CXX_FLAGS}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DGER_RUNTIME_OUTPUT_DIRECTORY})

include_directories(include)
include_directories(ext)

add_subdirectory(src)
