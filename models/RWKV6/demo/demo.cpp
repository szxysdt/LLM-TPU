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
struct rwkv_state {
  // rwkv state
  // TODO Determine the shape of this tensor
  std::vector<std::vector<int>> state_list;
};

class RWKV6 {
  // External Interface running on BM1684X
 public:
  void init(const std::vector<int> &devid, std::string model_path,
            std::string tokenizer_path);
  void deinit();
  // Normal chat interface
  void chat_rwkv();
  // IOT chat Interface
  void iot_chat_rwkv();

  // Internal implementation
 private:
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

 private:
  int device_num;
  bm_handle_t bm_handle;
  std::vector<bm_handle_t> handles;
  void *p_bmrt;

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

/**
 *
 */
void RWKV6::init(const std::vector<int> &devid, std::string model_path,
                 std::string tokenizer_path) {
  load_rwkv_tokenizer(tokenizer_path);
  return;
}

/**
 * RWKV forward process
 * Input:
 *  state+token
 * Output:
 * state+logits
 */
void RWKV6::rwkv_forward() { return; }

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

  std::vector<std::string> res_str;
  res_str = rwkv_tokenizer.decode(res);
  std::cout << "\r\ntoken2str=[";
  for (const auto &inner_str : res_str) {
    std::cout << inner_str << " ";
  }
  std::cout << "]\r\n" << std::endl;
}
void RWKV6::deinit() {
  // TODO bm_free_device && bmrt_destroy && bm_dev_free

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
  std::string model_path = "../models/llama2-7b_int4_1dev.bmodel";
  std::string tokenizer_path = "../support/tokenizer.model";
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
