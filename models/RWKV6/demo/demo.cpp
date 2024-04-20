//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "demo.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>

#include "memory.h"

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
  int num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = num_nets - 2;

  // TODO: visited_tokens ?
  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++) {
    std::string block_name = "block_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
  }

  // This API is not available, do not use it!
  // STATE_SIZE_1 = net_blocks[0]->stages[0].input_shapes->dims[1];
  // STATE_SIZE_2 = net_blocks[0]->stages[0].input_shapes->dims[2];
  
  // get state size
  STATE_SIZE_1 = 1584;
  STATE_SIZE_2 = 2048;

  // TODO: state cache

  return;
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
  printf("Demo for RWKV6 in BM1684X\n");
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
