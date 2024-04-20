#ifndef _RWKV_TOKENIZER_HPP_
#define _RWKV_TOKENIZER_HPP_

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/prettywriter.h"  // for stringify JSON
#include "rapidjson/stringbuffer.h"

// TODO 性能优化(多线程、异步？)

class Trie;

struct add_val {
  uint32_t idx;
  std::string token_str;
};

struct Result {
  size_t idx;
  Trie *u;
  std::unordered_map<uint32_t, std::string> values;
  // Constructor
  Result(size_t idx, Trie *u, std::unordered_map<uint32_t, std::string> values)
      : idx(idx), u(u), values(values) {}
  // Default constructor
  Result() : idx(0), u(nullptr), values() {}
};

class Trie {
 public:
  Trie(Trie *front = nullptr, u_char ch = 0);
  Trie *front;
  Trie *add(const std::string &key, add_val &val, size_t idx = 0);
  Result find_longest(std::string &key, size_t idx = 0);
  u_char ch;
  std::vector<Trie *> to;
  std::unordered_map<uint32_t, std::string> values;
};

class RWKV_Tokenizer {
 public:
  int load(const std::string &tokenizer_path);
  // RWKV_Tokenizer(const std::string &tokenizer_path);

  // 编码
  std::vector<std::vector<uint32_t>> encode(std::vector<std::string> &inputs);
  std::vector<std::vector<uint32_t>> encode(std::string &inputs);
  // 解码
  std::vector<std::string> decode(std::vector<std::vector<uint32_t>> &tokens);

 private:
  std::unordered_map<int, std::vector<uint8_t>> idx2token;
  // std::unordered_map<std::vector<std::string>, int> token2idx;
  std::vector<std::vector<uint8_t>> sorted;
  Trie root;

  // 单次编码: 字符->token
  std::vector<uint32_t> encodeBytes(std::string &inputs);
  // 单次解码: token->字符
  std::string decodeBytes(std::vector<uint32_t> &tokens);
};


#endif  // _RWKV_TOKENIZER_HPP_