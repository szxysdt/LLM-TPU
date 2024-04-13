#!/bin/bash
set -ex
models=
mode="f16"
folder="tmp"
num_device=1
mode_args=""
device_args=""
quantize_args="--quantize BF16"
name=""
num_layers=
out_model=$name.bmodel

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --mode)
            mode="$2"
            shift 2
            ;;
        --num_device)
            num_device="$2"
            shift 2
            ;;
        --name)
            name="$2"
            shift 2
            ;;
        *)
            echo "Invalid option: $key" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

#if [ "$name" = "rwkv6-1b5" ]; then
#  num_layers=31
#  echo "Compile RWKV6-1B5"
#elif [ "$name" = "rwkv6-3b" ]; then
#  num_layers=39
#  echo "Compile RWKV6-3B"
#else
#  >&2 echo -e "Error: Invalid name $name, the input name must be \033[31mrwkv6-1b5|rwkv6-3b\033[0m"
#  exit 1
#fi

if [ x$mode == x"bf16" ]; then
    quantize_args="--quantize BF16"
else
    echo "Error, unknown quantize mode (Now only support BF16)"
    exit 1
fi

if [ x$num_device != x1 ]; then
    device_args="--num_device $num_device"
    out_model=$name'_'$mode'_'$num_device'dev.bmodel'
else
    out_model=$name'_'$mode'_1dev.bmodel'
fi


outdir=${folder}/rwkv_mlir_model
mkdir -p $outdir
pushd $outdir

# Make MLIR
model_transform.py \
    --model_name rwkv \
    --model_def ../onnx/rwkv.onnx \
    --input_shapes [[1,1],[1,1584,2048]] \
    --mlir rwkv.mlir
# Make BMModel
model_deploy.py \
  --mlir rwkv.mlir \
  --quantize BF16 \
  --quant_input \
  --quant_output \
  --chip bm1684x \
  --model rwkv.bmodel


# It seems like...? It is not necessary to export the cache model
# # Make MLIR
# model_transform.py \
#     --model_name rwkv_cache \
#     --model_def ../onnx/rwkv.onnx \
#     --input_shapes [[1,1],[1,1584,2048]] \
#     --mlir rwkv_cache.mlir
# # Make BMModel
# model_deploy.py \
#   --mlir rwkv_cache.mlir \
#   --quantize BF16 \
#   --quant_input \
#   --quant_output \
#   --chip bm1684x \
#   --model rwkv_cache.bmodel

rm *.npz

models=$models' '$outdir'/rwkv.bmodel '

popd

echo $models


model_tool --combine $models -o $out_model