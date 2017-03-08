#!/bin/bash
clang --std=c++14 spmd/*.cpp t.cpp -mavx2 -mfma -lc++ -o t
