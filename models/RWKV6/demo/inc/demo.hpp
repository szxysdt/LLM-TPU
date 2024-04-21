#ifndef _DEMO_HPP_
#define _DEMO_HPP_

#include <assert.h>
#include <getopt.h>

#include <iostream>
#include <vector>

#include "bmruntime_interface.h"
#include "rwkv_tokenizer.hpp"

// struct rwkv_state {
//   // rwkv state
//   // TODO Determine the shape of this tensor
//   std::vector<std::vector<uint32_t>> state_list;
// };

class RWKV6 {
  // External Interface running on BM1684X
 public:
  void init(const std::vector<int> &devices, std::string model_path,
            std::string tokenizer_path, const std::string &__generation_mode);
  void deinit();
  // Normal chat interface
  void chat_rwkv();
  // IOT chat Interface
  // void iot_chat_rwkv();

  // Internal implementation
 private:
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);
  void net_launch(const bm_net_info_t *net, uint32_t stage_idx = 0);
  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  int greedy_search(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  // int penalty_sample(const bm_net_info_t *net, bm_device_mem_t &logits_mem);

  /**
   * Preprocessing of forward processes
   */
  void generate(const std::string &input_str);
  /**
   * RWKV forward process
   */
  int rwkv_forward_prefill();
  int rwkv_forward_rnn(uint32_t input_token);
  /**
   * Store the innner state of RWKV
   */
  // void store_state();
  /**
   * Format prompt template
   */
  // std::string format_prompt();
  /**
   * Load tokenizer
   */
  void load_rwkv_tokenizer(std::string tokenizer_path);

 public:
  int NUM_LAYERS = 0;
  std::string generation_mode;
  int max_gen_length = 100;  // for test

 private:
  //  模型句柄
  int device_num = 0;
  bm_handle_t bm_handle;
  std::vector<bm_handle_t> handles;
  void *p_bmrt;

  // 模型参数
  int STATE_SIZE_1 = 0;  // RWKV状态大小
  int STATE_SIZE_2 = 0;  // RWKV状态大小
  std::vector<const bm_net_info_t *> net_blocks;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_lm_head;
  const bm_net_info_t *net_greedy_head;

  // 内部变量
  const uint16_t STATE_INIT_DATA = 0x0000;
  std::vector<std::vector<uint32_t>> tokens_temp;  // for test
  // std::vector<bm_device_mem_t> state;
  // std::vector<std::pair<std::string, std::string>>      history_vector;  //
  // Temporarily not enabled

  // std::string sys_config = "hello";  // Temporarily not enabled

  // rwkv_tokenizer
  //
  RWKV_Tokenizer rwkv_tokenizer;

  // 初始化内存分配
  // 模型状态
  // std::vector<bm_tensor_t> rwkv_state_input;   // 输入状态
  // std::vector<bm_tensor_t> rwkv_state_output;  // 模型状态输出
  // std::vector<bm_tensor_t> rwkv_state_cache;   // 状态缓存

  // 模型输入token
  // std::vector<bm_tensor_t> rwkv_input_token;  // 输入token缓存

  // std::vector<bm_tensor_t> rwkv_output_logts;
};

#endif
