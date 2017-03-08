#!/bin/bash
clang --std=c++14 spmd/*.cpp t.cpp -mavx2 -lc++ -o t
