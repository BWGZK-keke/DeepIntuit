#!/usr/bin/env bash

set -ex
cd $(dirname $(realpath $0))

# 将模型文件先进行打包 tar -cvf xxxx.tar *， 然后放进 model_file
# 这里将压缩的文件进行解压，方便使用
tar -xvf model_file/*.tar -C  model_file/
wait