cmake_minimum_required(VERSION 3.13)
project(kmeans-openmpi)

set(CMAKE_CXX_STANDARD 14)

add_executable(kmeans-openmpi k_means_single.c)