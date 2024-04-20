#include "demo.hpp"
#include "untils.h"

void RWKV6::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  auto status = bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0,
                                   bm_mem_get_device_size(src));
  assert(status == BM_SUCCESS);
}

/**
 * io:  0=dump_bf16_tensor
 *      1=dump_fp16_tensor
 *      2=dump_fp32_tensor
 *      3=dump_int_tensor
 */
void RWKV6::net_launch(const bm_net_info_t *net, uint32_t stage_idx) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  for (int i = 0; i < net->input_num; i++) {
    bmrt_tensor_with_device(
        &in_tensors[i], net->stages[stage_idx].input_mems[i],
        net->input_dtypes[i], net->stages[stage_idx].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[stage_idx].output_mems[i],
        net->output_dtypes[i], net->stages[stage_idx].output_shapes[i]);
  }
  bool ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

void RWKV6::head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  bmrt_tensor_with_device(&in_tensors[0], logits_mem, net->input_dtypes[0],
                          net->stages[0].input_shapes[0]);

  for (int i = 1; i < net->input_num; i++) {
    bmrt_tensor_with_device(&in_tensors[i], net->stages[0].input_mems[i],
                            net->input_dtypes[i],
                            net->stages[0].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++) {
    bmrt_tensor_with_device(&out_tensors[i], net->stages[0].output_mems[i],
                            net->output_dtypes[i],
                            net->stages[0].output_shapes[i]);
  }
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

int RWKV6::greedy_search(const bm_net_info_t *net,
                         bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];

  head_launch(net, logits_mem);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

/**
 * RWKV forward process
 * Input:
 *  state+token
 * Output:
 * state+logits
 */
void RWKV6::rwkv_forward() {
  std::vector<uint16_t> state_init_data(STATE_SIZE_1 * STATE_SIZE_2,
                                        STATE_INIT_DATA);

  std::vector<uint32_t> input_data(tokens_temp[0].begin(),
                                   tokens_temp[0].end());

  uint32_t past_token;  // 上次推理出来的最后一个token
  for (int input_idx = 0; input_idx < input_data.size(); input_idx++) {
    uint32_t innner_inputdata = input_data[input_idx];

    // 没有状态/放弃前面的状态，则初始化状态
    // if (state_init_flag)...
  }
  // 准备输入数据
  // TODO 这里之后写个for i < tokens_temp.size() .....用于推理整个缓存
  // uint32_t input_data = 0;

  // input_data[0] = tokens_temp[0][0];
  // 一次推理一个token
  // input_data = tokens_temp[0][0];

  // std::cout << "input_data " << input_data << std::endl;
  // 初始化emb的输入内存
  bm_device_mem_t &emb_in_mem = net_embed->stages[0].input_mems[0];
  bm_device_mem_t &emb_out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, emb_in_mem, (void *)input_data.data());
  net_launch(net_embed);
  dump_int_tensor(bm_handle, emb_in_mem, 0, 1);

  // TODO get embedding dim in init (replace 2048)
  std::vector<uint16_t> test_cache(2048, 0);
  std::vector<uint16_t> test_cache2(2048, 0);
  bm_memcpy_d2s(bm_handle, (void *)test_cache.data(), emb_out_mem);
  for (size_t i = 0; i < test_cache.size(); i++) {
    test_cache2[i] = fp32_to_bf16_bits(fp16_ieee_to_fp32_bits(test_cache[i]));
  }

  // 循环外提前分配第一个block的输入mem

  bm_device_mem_t &out0_mem = net_blocks[0]->stages[0].output_mems[0];
  bm_device_mem_t &out1_mem = net_blocks[0]->stages[0].output_mems[1];
  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    bm_device_mem_t &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    bm_device_mem_t &in1_mem = net_blocks[idx]->stages[0].input_mems[1];

    if (idx == 0) {
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)state_init_data.data());
      bm_memcpy_s2d(bm_handle, in0_mem, (void *)test_cache2.data());

    } else {
      d2d(in0_mem, out0_mem);
      d2d(in1_mem, out1_mem);
    }
    net_launch(net_blocks[idx]);
    out0_mem = net_blocks[idx]->stages[0].output_mems[0];
    out1_mem = net_blocks[idx]->stages[0].output_mems[1];
  }
  // TODO 实现状态缓存和RNN推理

  // 映射最后block的输出
  bm_device_mem_t &out_mem =
      net_blocks[NUM_LAYERS - 1]->stages[0].output_mems[0];
  // bm_device_mem_t &state_mem =
  //     net_blocks[NUM_LAYERS - 1]->stages[0].output_mems[1];
  // lm_head
  bm_device_mem_t &lm_in_mem = net_lm_head->stages[0].input_mems[0];
  bm_device_mem_t &lm_out_mem = net_lm_head->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm_head);
  // dump tensor
  dump_fp32_tensor(bm_handle, lm_out_mem, 0, 50);

  // sample logits
  uint32_t token = 0;
  if (generation_mode == "greedy") {
    token = greedy_search(net_greedy_head, lm_out_mem);
    // } else if (generation_mode == "penalty_sample") {
    //   token = penalty_sample(net_penalty_sample_head, lm_out_mem);
  } else {
    std::cerr << "\nError: Invalid generation mode.\n";
    std::cerr << "Supported modes are 'greedy' or 'penalty_sample'.\n";
    throw std::runtime_error("Invalid generation mode");
  }
  std::cout << "token out = " << token << std::endl;

  return;
}
