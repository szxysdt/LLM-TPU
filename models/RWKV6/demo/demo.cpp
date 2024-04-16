//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <getopt.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "bmruntime_interface.h"
#include "memory.h"
#include "rwkv_tokenizer/rwkv_tokenizer.hpp"
#include "untils.h"

static const uint16_t STATE_INIT_DATA = 0x0000;

void print(auto info, auto input) {
  std::cout << info << "  " << input << std::endl;
}
void pprint(auto info, auto *input) {
  std::cout << info << "  " << *input << std::endl;
}

void dump_tensor(bm_handle_t bm_handle, bm_tensor_t &tensor) {
  auto shape = tensor.shape;
  int size = 1;
  for (int i = 0; i < shape.num_dims; ++i) {
    size *= shape.dims[i];
  }
  std::vector<uint16_t> data(size);
  bm_memcpy_d2s(bm_handle, (void *)data.data(), tensor.device_mem);
  std::cout << data[0] << "\t" << data[data.size() - 1] << std::endl;
  auto ptr = data.data();
  ptr[0] = ptr[0];
}

struct rwkv_state {
  // rwkv state
  // TODO Determine the shape of this tensor
  std::vector<std::vector<int>> state_list;
};

class RWKV6 {
  // External Interface running on BM1684X
 public:
  void init(const std::vector<int> &devices, std::string model_path,
            std::string tokenizer_path);
  void deinit();
  // Normal chat interface
  void chat_rwkv();
  // IOT chat Interface
  void iot_chat_rwkv();

  // Internal implementation
 private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);

  /**
   * Preprocessing of forward processes
   */
  void generate(const std::string &input_str);
  /**
   * RWKV forward process
   */
  void rwkv_forward();
  /**
   * Store the innner state of RWKV
   */
  void store_state();
  /**
   * Format prompt template
   */
  std::string format_prompt();
  /**
   * Load tokenizer
   */
  void load_rwkv_tokenizer(std::string tokenizer_path);

 public:
  int NUM_LAYERS;
  int STATE_SIZE_1, STATE_SIZE_2;  // RWKV状态大小

  std::vector<std::vector<uint32_t>> tokens_temp;  // for test

 private:
  int device_num;
  bm_handle_t bm_handle;
  std::vector<bm_handle_t> handles;
  void *p_bmrt;

  std::vector<const bm_net_info_t *> net_blocks;
  const bm_net_info_t *net_embed;
  const bm_net_info_t *net_lm_head;

  std::vector<bm_device_mem_t> state;
  std::vector<std::pair<std::string, std::string>>
      history_vector;  // Temporarily not enabled

  std::string sys_config = "hello";  // Temporarily not enabled

  // rwkv_tokenizer
  //
  RWKV_Tokenizer rwkv_tokenizer;

  // 初始化内存分配
  // 模型状态
  std::vector<bm_tensor_t> rwkv_state_input;   // 输入状态
  std::vector<bm_tensor_t> rwkv_state_output;  // 模型状态输出
  std::vector<bm_tensor_t> rwkv_state_cache;   // 状态缓存

  // 模型输入token
  std::vector<bm_tensor_t> rwkv_input_token;  // 输入token缓存

  std::vector<bm_tensor_t> rwkv_output_logts;
};

void RWKV6::d2d(bm_device_mem_t &dst, bm_device_mem_t &src) {
  auto status = bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0,
                                   bm_mem_get_device_size(src));
  assert(status == BM_SUCCESS);
}

/**
 * init rwkv model
 */
void RWKV6::init(const std::vector<int> &devices, std::string model_path,
                 std::string tokenizer_path) {
  // load tokenizer
  load_rwkv_tokenizer(tokenizer_path);

  // request bm_handle
  std::cout << "Device [ ";
  for (auto d : devices) {
    std::cout << d << " ";
  }
  std::cout << "] loading ....\n";
  for (auto d : devices) {
    bm_handle_t h;
    bm_status_t status = bm_dev_request(&h, d);
    assert(BM_SUCCESS == status);
    handles.push_back(h);
  }
  bm_handle = handles[0];

// create bmruntime
#ifdef SOC_TARGET
  p_bmrt = bmrt_create(handles[0]);
#else
  p_bmrt = bmrt_create_ex(handles.data(), handles.size());
#endif
  assert(NULL != p_bmrt);

  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  // get rwkv model
  net_embed = bmrt_get_network_info(p_bmrt, "embedding");
  net_lm_head = bmrt_get_network_info(p_bmrt, "lm_head");
  auto num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = num_nets - 2;

  // TODO: visited_tokens ?
  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    auto block_name = "block_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
  }

  // get state size
  STATE_SIZE_1 = net_blocks[0]->stages[0].input_shapes->dims[1];
  STATE_SIZE_2 = net_blocks[0]->stages[0].input_shapes->dims[2];

  // TODO: state cache

  return;
}

void RWKV6::net_launch(const bm_net_info_t *net, int stage_idx) {
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
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  auto status = bm_thread_sync(bm_handle);
  assert(status == BM_SUCCESS);
}

void print_dims(const bm_shape_t &shape) {
  std::cout << "num_dims: " << shape.num_dims << std::endl;
  std::cout << "dims: [";
  for (int i = 0; i < shape.num_dims; i++) {
    std::cout << shape.dims[i];
    if (i < shape.num_dims - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
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

  // 准备输入数据
  // TODO 这里之后写个for i < tokens_temp.size() .....用于推理整个缓存
  std::vector<int> input_data(tokens_temp[0].begin(), tokens_temp[0].end());

  // 初始化emb的输入内存
  auto &emb_in_mem = net_embed->stages[0].input_mems[0];
  auto &emb_out_mem = net_embed->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, emb_in_mem, (void *)input_data.data());
  net_launch(net_embed);

  // TODO get embedding dim in init (replace 2048)
  std::vector<uint16_t> test_cache(2048, 0);
  std::vector<uint16_t> test_cache2(2048, 0);
  bm_memcpy_d2s(bm_handle, (void *)test_cache.data(), emb_out_mem);
  for (int i = 0; i < test_cache.size(); i++) {
    test_cache2[i] = fp32_to_bf16_bits(fp16_ieee_to_fp32_bits(test_cache[i]));
  }

  // 循环外提前分配第一个block的输入mem
  auto &in0_mem = net_blocks[0]->stages[0].input_mems[0];
  bm_memcpy_s2d(bm_handle, in0_mem, (void *)test_cache2.data());

  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++) {
    // 分配输入映射
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];  // input emb
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];  // state
    // 分配输出映射
    // auto &out0_mem = net_blocks[idx]->stages[0].output_mems[0];  // output
    // emb auto &out1_mem = net_blocks[idx]->stages[0].output_mems[1];  // state

    // 拷贝数据
    if (idx == 0) {
      // TODO 把外面的fp32-bf16转换塞进循环？看情况而定...

      // 初始化状态
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)state_init_data.data());
      std::cout << "id == i copy ok" << std::endl;

    } else {
      // 映射上一个block的输出(output + state)
      auto &_out0_mem = net_blocks[idx - 1]->stages[0].output_mems[0];
      auto &_out1_mem = net_blocks[idx - 1]->stages[0].output_mems[1];
      auto aa_out1_mem = net_blocks[idx - 1]->stages[0].output_shapes->dims[0];
      std::cout << "_out1_mem " << aa_out1_mem << std::endl;

      // 从第二个block开始拷贝输出内容到输入
      d2d(in0_mem, _out0_mem);
      d2d(in1_mem, _out1_mem);
      std::cout << "copy ok id==" << idx << std::endl;
    }
    // dump inputs of a block

    // 开炮
    net_launch(net_blocks[idx]);
  }
  // TODO 实现状态缓存和RNN推理

  // 映射最后block的输出
  auto &out_mem = net_blocks[NUM_LAYERS - 1]->stages[0].output_mems[0];

  // lm_head
  auto &lm_in_mem = net_lm_head->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_head->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm_head);
  // dump tensor
  dump_fp32_tensor(bm_handle, lm_out_mem, 0, 50);

  // sample logits
  int token = 0;

  return;
}

void RWKV6::load_rwkv_tokenizer(std::string tokenizer_path) {
  std::cout << "Load Tokenizer " << tokenizer_path << " ..." << std::endl;
  auto status = rwkv_tokenizer.load(tokenizer_path);
  if (status) {
    std::cerr << "Load tokenizer error, please check your tokenizer" << status
              << std::endl;
    exit(-1);
  }
  // TODO rwkv eos??
  std::cout << "Done!\n" << std::endl;
}

void RWKV6::chat_rwkv() {
  while (true) {
    std::cout << "\nUser: ";
    std::string user_input;
    std::getline(std::cin, user_input);
    if (user_input == "exit") {
      break;
    }
    std::cout << "\nAssistant: " << std::flush;
    generate(user_input);
    std::cout << std::endl;
  }
}

void RWKV6::generate(const std::string &input_str) {
  //
  /** test code ***********************************/
  std::string input_str_ = input_str;
  std::vector<std::vector<uint32_t>> res;
  res = rwkv_tokenizer.encode(input_str_);
  // simple print
  std::cout << "\r\nstr2token=[";
  for (const auto &inner_vec : res) {
    std::cout << "[";
    for (uint32_t val : inner_vec) {
      std::cout << val << ", ";
    }
    std::cout << "]";
  }
  std::cout << "]\r\n" << std::endl;
  // copy token_ids2temp_cache
  tokens_temp.clear();
  std::copy(res.begin(), res.end(), std::back_inserter(tokens_temp));

  std::cout << "\r\ntokens_temp=[";
  for (const auto &inner_vec : tokens_temp) {
    std::cout << "[";
    for (uint32_t val : inner_vec) {
      std::cout << val << ", ";
    }
    std::cout << "]";
  }
  std::cout << "]\r\n" << std::endl;

  rwkv_forward();

  std::vector<std::string> res_str;
  res_str = rwkv_tokenizer.decode(res);
  std::cout << "\r\ntoken2str=[";
  for (const auto &inner_str : res_str) {
    std::cout << inner_str << " ";
  }
  std::cout << "]\r\n" << std::endl;

  /** test code ***********************************/
}
void RWKV6::deinit() {
  // TODO bm_free_device && bmrt_destroy && bm_dev_free
  // if (false == io_alone) {
  // for (int i = 0; i < NUM_LAYERS; i++) {
  //   bm_free_device(bm_handle, state[i]);
  // }
  // }
  bmrt_destroy(p_bmrt);
  for (auto h : handles) {
    bm_dev_free(h);
  }
  return;
}

// std::string RWKV6::format_prompt() {}

static void split(const std::string &s, const std::string &delim,
                  std::vector<std::string> &ret) {
  size_t last = 0;
  size_t index = s.find_first_of(delim, last);
  while (index != std::string::npos) {
    ret.push_back(s.substr(last, index - last));
    last = index + 1;
    index = s.find_first_of(delim, last);
  }
  if (last < s.length()) {
    ret.push_back(s.substr(last));
  }
}

static std::vector<int> parseCascadeDevices(const std::string &str) {
  std::vector<int> devices;
  std::vector<std::string> sub_str;
  split(str, ",", sub_str);
  for (auto &s : sub_str) {
    devices.push_back(std::atoi(s.c_str()));
  }
  return devices;
}

void Usage() {
  std::string help =
      "Usage:\n"
      "  --help         : Show help info.\n"
      "  --model        : Set model path \n"
      "  --tokenizer    : Set tokenizer path \n"
      "  --devid        : Set devices to run for model, e.g. 1,2. if not "
      "set, use 0\n";
  std::cout << help << std::endl;
}

void processArguments(int argc, char *argv[], std::string &model_path,
                      std::string &tokenizer_path, std::vector<int> &devices) {
  struct option longOptions[] = {{"model", required_argument, nullptr, 'm'},
                                 {"tokenizer", required_argument, nullptr, 't'},
                                 {"devid", required_argument, nullptr, 'd'},
                                 {"help", no_argument, nullptr, 'h'},
                                 {nullptr, 0, nullptr, 0}};

  int optionIndex = 0;
  int option;

  while ((option = getopt_long(argc, argv, "m:t:d:h:", longOptions,
                               &optionIndex)) != -1) {
    switch (option) {
      case 'm':
        model_path = optarg;
        break;
      case 't':
        tokenizer_path = optarg;
        break;
      case 'd':
        devices = parseCascadeDevices(optarg);
        break;
      case 'h':
        Usage();
        exit(EXIT_FAILURE);
      case '?':
        Usage();
        exit(EXIT_FAILURE);
      default:
        exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char **argv) {
  // set your bmodel path here
  printf("Demo for LLama2 in BM1684X\n");
  std::string model_path = "../models/rwkv6-1b5_bf16_1dev.bmodel";
  std::string tokenizer_path = "../support/rwkv_vocab_v20230424.json";
  std::vector<int> devices = {11};
  processArguments(argc, argv, model_path, tokenizer_path, devices);
  if (model_path.empty()) {
    Usage();
    exit(EXIT_FAILURE);
  }

  RWKV6 rwkv6;
  printf("Init Environment ...\n");
  rwkv6.init(devices, model_path, tokenizer_path);
  printf("==========================\n");
  rwkv6.chat_rwkv();
  rwkv6.deinit();
  return 0;
}
