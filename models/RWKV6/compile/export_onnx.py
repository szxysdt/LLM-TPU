# Reference from https://github.com/yuunnn-w/RWKV_Pytorch

import argparse
import os

import torch
# torch.set_printoptions(profile="full")
from tqdm import tqdm

from src.model import RWKV_RNN


class Embedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        emb_out = origin_model.emb(input_ids).squeeze(1)
        ln_out = origin_model.manual_layer_norm(
            emb_out, origin_model.ln0_weight, origin_model.ln0_bias, 1e-5
        ).unsqueeze(dim=0)
        return ln_out


class Block(torch.nn.Module):

    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.layer = origin_model.blocks[layer_id]

    def forward(self, b_in, state, b_id):
        b_out = self.layer(b_in, state, b_id)
        return b_out


class LmHead(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, b_out):
        head_in = origin_model.manual_layer_norm(
            b_out, origin_model.ln_out_weight, origin_model.ln_out_bias, 1e-5
        )
        m_logits = origin_model.head(head_in)
        return m_logits


def convert_block(layer_id, verbose=False):
    model = Block(layer_id)
    b_in = torch.zeros(model_args["batch_size"], 1, EMB_DIM)
    state = torch.randn(model_args["batch_size"], *STATE_SIZE)
    b_id = torch.tensor([layer_id], dtype=torch.int64)

    torch.onnx.export(
        model,
        (b_in, state, b_id),
        f"{folder}/block_{layer_id}.onnx",
        verbose=verbose,
        input_names=["b_in", "state", "b_id"],
        output_names=["b_out"],
        do_constant_folding=True,
        opset_version=15,
    )


def convert_embedding():
    model = Embedding()
    input_ids = torch.zeros(model_args["batch_size"], 1).long()

    torch.onnx.export(
        model,
        (input_ids),
        f"{folder}/embedding.onnx",
        verbose=False,
        input_names=["input_ids"],
        output_names=["hidden_dim"],
        do_constant_folding=True,
        opset_version=15,
    )


def test_emb():
    model = Embedding()
    input_ids = torch.tensor([[1922]]).long()
    out = model(input_ids)
    print(f"out {out}")


def convert_lm_head():
    model = LmHead()
    input = torch.randn(model_args["batch_size"], 1, EMB_DIM)

    torch.onnx.export(
        model,
        (input),
        f"{folder}/lm_head.onnx",
        verbose=False,
        input_names=["hidden_dim"],
        output_names=["m_logits"],
        do_constant_folding=True,
        opset_version=15,
    )


def test_lm_head():
    model = LmHead()
    input = torch.randn(model_args["batch_size"], 1, EMB_DIM)
    out = model(input)
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export onnx.")
    parser.add_argument("--model_path", "-m", type=str, help="path to the torch model.")
    parser.add_argument("--test", "-t", action="store_true", help="enable some tests.")
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

    NUM_LAYERS = origin_model.num_layer
    EMB_DIM = origin_model.n_embd
    print("EMB_DIM")
    print(EMB_DIM)
    STATE_SIZE = origin_model.state_size
    print(origin_model)
    print("Done.")

    origin_model.eval()  # 确保模型处于评估模式
    for param in origin_model.parameters():
        param.requires_grad = False

    # 准备输入数据的示例
    example_token = torch.zeros(
        model_args["batch_size"], 1
    ).long()  # token输入的尺寸 [batch, 1]
    example_state = torch.randn(
        model_args["batch_size"], *origin_model.state_size
    )  # state_size是state输入的尺寸
    # 测试推理
    A, B = origin_model(example_token, example_state)

    print(A)
    if args.test:
        test_emb()
        # test_lm_head()
        exit(0)

    # 导出模型
    print("\nExport Onnx...")

    print(f"Convert block & block_cache")
    for i in tqdm(range(NUM_LAYERS)):
        convert_block(i)

    print(f"Convert embedding")
    convert_embedding()

    print(f"Convert lm_head")
    convert_lm_head()

    print(f"\nDone.\nOnnx weight has saved in {folder}")
