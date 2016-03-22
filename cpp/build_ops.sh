#!/usr/bin/env bash

TF_INC=$(python2 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
mkdir lib
g++ -std=c++11 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I $TF_INC aggregator.cc -o lib/aggregator.so
