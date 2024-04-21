#include "demo.hpp"

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
  std::vector<std::vector<uint32_t>> res = rwkv_tokenizer.encode(input_str_);
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


// prefill 输出第一个token
  uint32_t token = rwkv_forward_prefill();
  std::string token_string = rwkv_tokenizer.decode(token);
  std::cout << "decode token= " << token_string << std::endl;
  
// while (token != 0 && token_length < max_gen_length) {
//     result_tokens.emplace_back(token);
//     token = rwkv_forward_rnn(token);
//   }


  std::vector<std::string> res_str;
  res_str = rwkv_tokenizer.decode(res);
  std::cout << "\r\ntoken2str=[";
  for (const auto &inner_str : res_str) {
    std::cout << inner_str << " ";
  }
  std::cout << "]\r\n" << std::endl;

  /** test code ***********************************/
}