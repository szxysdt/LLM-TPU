#!/bin/bash

# For testing , it will be removed
clear
cd demo &&  cd build
cmake .. && make clean && make -j8 || exit 1
cd ../..
echo $PWD

# ./demo/build/rwkv6 --model ../../bmodels/rwkv6-1b5_bf16_1dev.bmodel --tokenizer ./support/rwkv_vocab_v20230424.json --devid 0
# ./demo/build/rwkv6 --model /data/work/sd_card/rwkv6-1b5_bf16_1dev.bmodel --tokenizer ./support/rwkv_vocab_v20230424.json --devid 0

# ./demo/build/rwkv6 --model /data/disk/rwkv6-1b5_bf16_1dev.bmodel --tokenizer ./support/rwkv_vocab_v20230424.json --devid 0
./demo/build/rwkv6 --model /data/usb_disk/dds/rwkv6-1b5_bf16_1dev.bmodel --tokenizer ./support/rwkv_vocab_v20230424.json --devid 0
# ./demo/build/rwkv6 --model /data/usb_disk/dds/rwkv6-1b5_bf16_f32emb_1dev.bmodel --tokenizer ./support/rwkv_vocab_v20230424.json --devid 0
