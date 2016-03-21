#!/usr/bin/env bash

TF_INC=$(python2 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
mkdir lib
g++ -std=c++11 -shared -fPIC aggregator.cc -o lib/aggregator.so -I $TF_INC