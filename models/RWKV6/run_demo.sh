#!/bin/bash
# # download bmodel
# if [ ! -d "../../bmodels" ]; then
#   mkdir ../../bmodels
# fi

# if [ ! -f "../../bmodels/rwkv6-1b5_bf16_1dev.bmodel" ]; then
#   pip3 install dfss
#   python3 -m dfss --url=open@sophgo.com:/ext_model_information/LLM/LLM-TPU/rwkv6-1b5_bf16_1dev.bmodel
#   mv rwkv6-1b5_bf16_1dev.bmodel ../../bmodels
# else
#   echo "Bmodel Exists!"
# fi

# make the file
if [ ! -f "./demo/rwkv6" ]; then
  cd demo && rm -rf build && mkdir build && cd build
  cmake .. && make -j
  cp rwkv6 ..
  cd ../..
else
  echo "rwkv6 file Exists!"
fi


# run demo
echo $PWD
# ./demo/rwkv6 --model ../../bmodels/rwkv.bmodel --tokenizer ./support/rwkv_vocab_v20230424.json --devid 0
./demo/rwkv6 --model ../../bmodels/rwkv.bmodel --tokenizer ./support/rwkv_vocab_v20230424.json --devid 0
