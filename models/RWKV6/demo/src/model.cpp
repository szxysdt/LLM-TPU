#include "demo.hpp"
#include "untils.h"

void RWKV6::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  auto status = bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0,
                                   bm_mem_get_device_size(src));
  assert(status == BM_SUCCESS);
}

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
 * RWKV forward process (Prefill)
 * 执行前，将需要prefill的token填入tokens_temp
 *
 * 会根据tokens_temp输出下一个token，并且保留state_cache
 *
 */
int RWKV6::rwkv_forward_prefill() {
  std::vector<uint16_t> state_init_data(STATE_SIZE_1 * STATE_SIZE_2,
                                        STATE_INIT_DATA);

  std::vector<uint32_t> input_data(tokens_temp[0].begin(),
                                   tokens_temp[0].end());
  std::vector<uint32_t> innner_inputdata(1, 0x00000000);
  uint32_t output_token_temp = 0;  // 复用的token缓存

  // RNN模式进行prefill
  for (size_t input_idx = 0; input_idx < input_data.size(); input_idx++) {
    // 每次计算一个token
    innner_inputdata.clear();
    innner_inputdata[0] = input_data[input_idx];
    std::cout << "innner_inputdata " << innner_inputdata[0] << std::endl;
    /**
     * emb forward
     */
    // 初始化emb的输入内存
    bm_device_mem_t &emb_in_mem = net_embed->stages[0].input_mems[0];
    bm_device_mem_t &emb_out_mem = net_embed->stages[0].output_mems[0];
    // 输入单个token
    bm_memcpy_s2d(bm_handle, emb_in_mem, (void *)innner_inputdata.data());
    // dump_int_tensor(bm_handle, emb_in_mem, 0, 1);
    net_launch(net_embed);  // forward emb
    /**
     * blocks forward
     */
    // 初始化blocks内存映射
    bm_device_mem_t &out0_mem = net_blocks[0]->stages[0].output_mems[0];
    bm_device_mem_t &out1_mem = net_blocks[0]->stages[0].output_mems[1];
    // forward blocks
    for (int idx = 0; idx < NUM_LAYERS; idx++) {
      bm_device_mem_t &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
      bm_device_mem_t &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
      if (idx == 0) {
        /**
         * 第一层forward需要传入状态
         */
        // TODO 根据不同的模式，初始化不同的状态
        // 例如：没有状态/放弃前面的状态，则初始化状态
        // if (state_init_flag)...
        if (input_idx == 0) {
          // 初始化state为0
          bm_memcpy_s2d(bm_handle, in1_mem, (void *)state_init_data.data());
        } else {
          // 初始化state为上一波跑完的state
          bm_device_mem_t &past_out1_mem =
              net_blocks[NUM_LAYERS - 1]->stages[0].output_mems[1];
          d2d(in1_mem, past_out1_mem);
        }
        // 第一层，输入来自emb
        d2d(in0_mem, emb_out_mem);
      } else {
        // 非第一层，复制上一层输出
        out0_mem = net_blocks[idx - 1]->stages[0].output_mems[0];
        out1_mem = net_blocks[idx - 1]->stages[0].output_mems[1];
        d2d(in0_mem, out0_mem);
        d2d(in1_mem, out1_mem);
      }
      // start forward
      net_launch(net_blocks[idx]);
    }

    // prefill不用跑head，输出才·跑
  }

  // 重新分配输出mem映射
  bm_device_mem_t &out_mem =
      net_blocks[NUM_LAYERS - 1]->stages[0].output_mems[0];

  bm_device_mem_t &lm_in_mem = net_lm_head->stages[0].input_mems[0];
  bm_device_mem_t &lm_out_mem = net_lm_head->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm_head);

  // sample logits
  if (generation_mode == "greedy") {
    output_token_temp = greedy_search(net_greedy_head, lm_out_mem);
    // } else if (generation_mode == "penalty_sample") {
    //   token = penalty_sample(net_penalty_sample_head, lm_out_mem);
  } else {
    std::cerr << "\nError: Invalid generation mode.\n";
    std::cerr << "Supported modes are 'greedy' or 'penalty_sample'.\n";
    throw std::runtime_error("Invalid generation mode");
  }

  return output_token_temp;
}
/**
 * RWKV forward process (RNN)
 * 执行前，需要进行prefill，生成前文的state_cache(存放最后一个block的out_mem内)
 *
 *
 */
int RWKV6::rwkv_forward_rnn(uint32_t input_token) {
  std::vector<uint32_t> innner_inputdata(1, 0x00000000);
  uint32_t output_token_temp = 0;  // 复用的token缓存

  innner_inputdata[0] = input_token;  // 填充输入token

  /**
   * emb forward
   */
  // 初始化emb的输入内存
  bm_device_mem_t &emb_in_mem = net_embed->stages[0].input_mems[0];
  bm_device_mem_t &emb_out_mem = net_embed->stages[0].output_mems[0];
  // 输入单个token
  bm_memcpy_s2d(bm_handle, emb_in_mem, (void *)innner_inputdata.data());
  // dump_int_tensor(bm_handle, emb_in_mem, 0, 1);
  net_launch(net_embed);  // forward emb
  /**
   * blocks forward
   */
  bm_device_mem_t &out0_mem = net_blocks[0]->stages[0].output_mems[0];  // 输入
  bm_device_mem_t &out1_mem = net_blocks[0]->stages[0].output_mems[1];
  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    bm_device_mem_t &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    bm_device_mem_t &in1_mem = net_blocks[idx]->stages[0].input_mems[1];

    if (idx == 0) {
      // 第一层forward需要传入状态
      // TODO 根据不同的模式，初始化不同的状态
      // 例如：没有状态/放弃前面的状态，则初始化状态
      // if (state_init_flag)...

      // RNN生成模式，初始化state为上一波跑完的state
      bm_device_mem_t &past_out1_mem =
          net_blocks[NUM_LAYERS - 1]->stages[0].output_mems[1];
      d2d(in1_mem, past_out1_mem);
      // 第一层，输入来自emb
      d2d(in0_mem, emb_out_mem);
    } else {
      // 非第一层，复制上一层输出
      out0_mem = net_blocks[idx - 1]->stages[0].output_mems[0];
      out1_mem = net_blocks[idx - 1]->stages[0].output_mems[1];
      d2d(in0_mem, out0_mem);
      d2d(in1_mem, out1_mem);
    }
    // start forward
    net_launch(net_blocks[idx]);
  }
  // 重新分配输出mem映射
  bm_device_mem_t &out_mem =
      net_blocks[NUM_LAYERS - 1]->stages[0].output_mems[0];

  bm_device_mem_t &lm_in_mem = net_lm_head->stages[0].input_mems[0];
  bm_device_mem_t &lm_out_mem = net_lm_head->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm_head);

  // sample logits
  if (generation_mode == "greedy") {
    output_token_temp = greedy_search(net_greedy_head, lm_out_mem);
    // } else if (generation_mode == "penalty_sample") {
    //   token = penalty_sample(net_penalty_sample_head, lm_out_mem);
  } else {
    std::cerr << "\nError: Invalid generation mode.\n";
    std::cerr << "Supported modes are 'greedy' or 'penalty_sample'.\n";
    throw std::runtime_error("Invalid generation mode");
  }
  return output_token_temp;
}
