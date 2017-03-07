#!/bin/bash
clang --std=c++11 spmd.cpp cpu.cpp -mavx2 -o spmd -lc++
