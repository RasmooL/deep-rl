#!/usr/bin/env bash

TF_INC=$(python2 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
if [ ! -d "lib" ]; then
  mkdir lib
fi
for file in *.cc; do
  g++   -Wall -Werror -Wno-unused -Wno-sign-compare\
        -O2 -std=c++11 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -I ${TF_INC} \
        "$file" -o "lib/${file/.cc/.so}" &
done