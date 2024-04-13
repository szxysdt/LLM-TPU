# Reference from https://github.com/yuunnn-w/RWKV_Pytorch

import argparse
import os

import torch

from src.model import RWKV_RNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export onnx.")
    parser.add_argument("--model_path", "-m", type=str, help="path to the torch model.")
    # parser.add_argument('--model_path', type=str, help='path to the torch model.')
    # parser.add_argument('--seq_length', type=int, default=512, help="sequence length")

    args = parser.parse_args()

    model_path = args.model_path
    folder = f"./tmp/onnx"
    # create folder to store onnx
    if not os.path.exists(folder):
        os.makedirs(folder)

    model_args = {
        "MODEL_NAME": model_path,  # 模型文件的名字，pth结尾的权重文件。
        "vocab_size": 65536,  # 词表大小
        "batch_size": 1,
    }
    print(f"Loading model {model_args['MODEL_NAME']}.pth...")
    origin_model = RWKV_RNN(model_args)
    print(origin_model)
    print("Done.")

    origin_model.eval()  # 确保模型处于评估模式
    for param in origin_model.parameters():
        param.requires_grad = False

    # 准备输入数据的示例
    example_token = torch.zeros(
        model_args["batch_size"], 1
    ).long()  # token输入的尺寸 [batch, 1]
    example_state = torch.rand(
        model_args["batch_size"], *origin_model.state_size
    )  # state_size是state输入的尺寸
    # print("Example token shape:", example_token.shape)
    # print("Example state shape:", example_state.shape)
    # 测试推理
    A, B = origin_model(example_token, example_state)
    # 导出模型
    print("\nExport Onnx...")

    torch.onnx.export(
        origin_model,
        (example_token, example_state),
        f"{folder}/rwkv.onnx",
        export_params=True,
        verbose=True,
        opset_version=12,  # LayerNorm最低支持是op17
        do_constant_folding=True,
        input_names=["token", "input_state"],
        output_names=["out", "out_state"],
        dynamic_axes={
            "token": {0: "batch_size"},
            "input_state": {0: "batch_size"},
            "out": {0: "batch_size"},
            "out_state": {0: "batch_size"},
        },
    )

    print(f"\nDone.\nOnnx weight has saved in {folder}/rwkv.onnx")
