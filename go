#!/bin/bash
clang --std=c++11 spmd.cpp spmd/cpu.cpp spmd/avx.cpp -mavx2 -o spmd -lc++
