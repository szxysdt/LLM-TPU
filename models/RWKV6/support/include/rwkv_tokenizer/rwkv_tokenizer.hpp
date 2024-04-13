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

const char *kTypeNames[] = {"Null",  "False",  "True",  "Object",
                            "Array", "String", "Number"};

Trie::Trie(Trie *front, u_char ch) : front(front), ch(ch) {
  to = std::vector<Trie *>(256, nullptr);
};

/**
 * key=字符
 * idx=树序号
 */
Trie *Trie::add(const std::string &key, add_val &val, size_t idx) {
  if (idx == key.size()) {
    add_val add_value = val;
    values[add_value.idx] = add_value.token_str;
    return this;
  }
  u_char ch = key[idx];
  if (to[ch] == nullptr) {
    to[ch] = new Trie(this, ch);
  }
  return to[ch]->add(key, val, idx + 1);
};

/**
 * 检索
 */
Result Trie::find_longest(std::string &key, size_t idx) {
  Trie *u = this;
  u_char ch = key[idx];
  //   std::cout << "ch " << ch << std::endl;

  Result ret = {idx = idx, u, {}};
  while (u->to[ch] != nullptr) {
    u = u->to[ch];
    idx += 1;

    if (!u->values.empty()) {
      ret = Result{idx, u, u->values};
    }
    //
    if (idx == key.size()) {
      break;
    }
    ch = key[idx];
  }
  return ret;
}

int RWKV_Tokenizer::load(const std::string &tokenizer_path) {
  FILE *fp = fopen(tokenizer_path.c_str(), "r");
  char readBuffer[65536];
  rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
  rapidjson::Document d;
  d.ParseStream(is);
  fclose(fp);
  if (d.HasParseError()) {
    std::cout << "Error parsing JSON" << std::endl;
    return 1;
  }
  if (d.IsObject()) {
    // 遍历 JSON 对象的所有成员
    for (rapidjson::Value::ConstMemberIterator itr = d.MemberBegin();
         itr != d.MemberEnd(); ++itr) {
      std::vector<uint8_t> value;
      int key = std::stoi(itr->name.GetString());  // 获取键

      if (itr->value.IsString()) {
        std::vector<uint8_t> result;
        size_t i = 0;

        while (i < itr->value.GetStringLength()) {
          std::string value = {itr->value.GetString()};
          uint8_t cc = value[i];
          result.push_back(cc);
          i += 1;
        }
        idx2token[key] = result;

      } else if (itr->value.IsArray()) {
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        itr->value.Accept(writer);
        auto test_array = itr->value.GetArray();

        if (itr->value.IsArray()) {
          for (size_t i = 0; i < (itr->value.Size()); i++) {
            if (itr->value[i].IsUint()) {
              uint8_t _value = static_cast<uint8_t>(itr->value[i].GetUint());
              value.push_back(_value);
              // std::cout << "【" << static_cast<int>(_value) << "】 ";
            } else {
              std::cerr << "数据类型错误\n";
              return 1;
            }
          }
        }

        idx2token[key] = value;
      } else {
        printf("Type of member is %s\n", kTypeNames[itr->value.GetType()]);
        // auto type = itr->value.GetType();
      };
    }

    for (const auto &entry : idx2token) {
      uint32_t key = entry.first;
      const std::vector<uint8_t> &value = entry.second;
      // 将value中的数据组合成一个字符串
      std::string value_str;
      for (const auto &str : value) {
        value_str += str;
      }
      add_val add_v;
      add_v.idx = key;
      add_v.token_str = value_str;
      // add to trie
      root.add(value_str, add_v);
    }
    // Normal return
    return 0;
  }
  return 1;
};

std::vector<uint32_t> RWKV_Tokenizer::encodeBytes(std::string &inputs) {
  size_t idx = 0;
  std::vector<uint32_t> tokens;
  while (idx < inputs.size()) {
    Result res = root.find_longest(inputs, idx);
    assert(res.idx != idx);
    auto it = std::begin(res.values);
    uint32_t token = it->first;
    tokens.push_back(token);
    idx = res.idx;
  }
  return tokens;
};

std::string RWKV_Tokenizer::decodeBytes(std::vector<uint32_t> &tokens) {
  std::string result;
  for (auto token : tokens) {
    // 查找token对应的字符串向量
    std::string it(this->idx2token[token].begin(),
                   this->idx2token[token].end());
    result += it;
  }
  return result;
};

std::vector<std::vector<uint32_t>> RWKV_Tokenizer::encode(std::string &input) {
  std::vector<std::vector<uint32_t>> result;
  result.push_back(
      encodeBytes(input));  // 将编码后的结果作为唯一的元素添加到结果向量中
  return result;
}
// 处理字符串列表输入的 encode 函数
std::vector<std::vector<uint32_t>> RWKV_Tokenizer::encode(
    std::vector<std::string> &inputs) {
  std::vector<std::vector<uint32_t>> result;
  for (std::string &input : inputs) {
    result.push_back(
        encodeBytes(input));  // 对每个字符串进行编码并添加到结果向量中
  }
  return result;
}
// 解码函数
std::vector<std::string> RWKV_Tokenizer::decode(
    std::vector<std::vector<uint32_t>> &tokens) {
  std::vector<std::string> decodedStrings;
  for (auto &batch : tokens) {
    decodedStrings.push_back(
        decodeBytes(batch));  // 解码每一批tokens并添加到结果向量中
  }
  return decodedStrings;
}

#endif  // RWKV_TOKENIZER_HPP