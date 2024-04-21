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
                 std::string tokenizer_path,
                 const std::string &__generation_mode) {
  // load tokenizer
  load_rwkv_tokenizer(tokenizer_path);

  // init settings
  generation_mode = __generation_mode;

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
  net_greedy_head = bmrt_get_network_info(p_bmrt, "greedy_head");
  int num_nets = bmrt_get_network_number(p_bmrt);
  NUM_LAYERS = num_nets - 3;

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
  int state_byte_size_1 = net_blocks[0]->max_input_bytes[0];
  int state_byte_size_2 = net_blocks[0]->max_input_bytes[1];
  int first_state_size = state_byte_size_2 / state_byte_size_1;
  int second_state_size = net_embed->stages[0].output_shapes->dims[1];
  STATE_SIZE_1 = first_state_size;
  STATE_SIZE_2 = second_state_size;

  // TODO: state cache (bm tensor)

  return;
}

void RWKV6::deinit() {
  // TODO bm_free_device && bmrt_destroy && bm_dev_free
  // if (false == io_alone) {
  //   bm_free_device(bm_handle, state);
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
  printf(
      "Usage:\n"
      "  --help                  : Show help info.\n"
      "  --model                 : Set model path \n"
      "  --tokenizer             : Set tokenizer path \n"
      "  --devid                 : Set devices to run for model, e.g. 1,2, if "
      "not provided, use 0\n"
      "  --temperature           : Set temperature for generating new token, "
      "e.g. 1.0, if not provided, default to 1.0 \n"
      "  --top_p                 : Set top_p for generating new tokens, e.g. "
      "0.8, if not provided, default to 1 \n"
      "  --repeat_penalty        : Set repeat_penalty for generating new "
      "tokens, e.g. 1.1, if not provided, default to 1.1 \n"
      "  --repeat_last_n         : Set repeat_penalty for penalizing recent n "
      "tokens, e.g. 32, if not provided, default to 32 \n"
      "  --max_new_tokens        : Set max new tokens, e.g. 100, if not "
      "provided, stop at EOS or exceeding max length \n"
      "  --generation_mode       : Set generation mode, e.g sample in greedy "
      "or penalty_sample, if not provided, default to greedy search \n"
      "  --input_mode            : Set input mode, e.g. unprompted, if not "
      "provided, use prompted \n"
      "\n");
}

void processArguments(int argc, char *argv[], std::string &model_path,
                      std::string &tokenizer_path, std::vector<int> &devices,
                      float &temperature, uint16_t &top_p,
                      float &repeat_penalty, int &repeat_last_n,
                      int &max_new_tokens, std::string &generation_mode,
                      std::string &input_mode) {
  struct option longOptions[] = {
      {"model", required_argument, nullptr, 'm'},
      {"tokenizer", required_argument, nullptr, 't'},
      {"devid", required_argument, nullptr, 'd'},
      {"help", no_argument, nullptr, 'h'},
      {"temperature", required_argument, nullptr, 'e'},
      {"top_p", required_argument, nullptr, 'p'},
      {"repeat_penalty", required_argument, nullptr, 'r'},
      {"repeat_last_n", required_argument, nullptr, 'l'},
      {"max_new_tokens", required_argument, nullptr, 'n'},
      {"generation_mode", required_argument, nullptr, 'g'},
      {"input_mode", required_argument, nullptr, 'i'},
      {nullptr, 0, nullptr, 0}};

  int optionIndex = 0;
  int option;

  while ((option = getopt_long(argc, argv, "m:t:d:h:e:p:r:l:n:g", longOptions,
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
      case 'e':
        temperature = std::stof(optarg);
        break;
      case 'p':
        top_p = std::stof(optarg);
        break;
      case 'r':
        repeat_penalty = std::stof(optarg);
        break;
      case 'l':
        repeat_last_n = std::stoi(optarg);
        break;
      case 'n':
        max_new_tokens = std::stoi(optarg);
        break;
      case 'g':
        generation_mode = optarg;
        break;
      case 'i':
        input_mode = optarg;
        break;
      case 'h':
        Usage();
        exit(EXIT_SUCCESS);
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
  std::string tokenizer_path = "./rwkv_tokenizer/rwkv_vocab_v20230424.json";
  std::vector<int> devices = {1};
  float temperature = 1.f;
  uint16_t top_p = 1;
  float repeat_penalty = 1.1f;
  int repeat_last_n = 32;
  int max_new_tokens = std::numeric_limits<int>::max();
  std::string generation_mode = "greedy";
  std::string input_mode = "prompted";
  processArguments(argc, argv, model_path, tokenizer_path, devices, temperature,
                   top_p, repeat_penalty, repeat_last_n, max_new_tokens,
                   generation_mode, input_mode);
  if (model_path.empty()) {
    Usage();
    exit(EXIT_FAILURE);
  }

  RWKV6 rwkv6;
  printf("Init Environment ...\n");
  rwkv6.init(devices, model_path, tokenizer_path, generation_mode);
  printf("==========================\n");
  rwkv6.chat_rwkv();
  rwkv6.deinit();
  return 0;
}
