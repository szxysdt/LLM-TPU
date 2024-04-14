#!/bin/bash

# For testing , it will be removed

cd demo &&  cd build
cmake .. && make -j || exit 1
cd ../..
echo $PWD

./demo/build/rwkv6 --model ../../bmodels/rwkv6-1b5_bf16_1dev.bmodel --tokenizer ./support/rwkv_vocab_v20230424.json --devid 0
