#!/bin/bash
clang --std=c++11 spmd/*.cpp t.cpp -mavx2 -lc++ -o t
