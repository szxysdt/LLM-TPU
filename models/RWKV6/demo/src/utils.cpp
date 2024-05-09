#include "demo.hpp"


void RWKV6::load_rwkv_tokenizer(std::string tokenizer_path) {
  std::cout << "Load Tokenizer " << tokenizer_path << " ..." << std::endl;
  int status = rwkv_tokenizer.load(tokenizer_path);
  if (status) {
    std::cerr << "Load tokenizer error, please check your tokenizer" << status
              << std::endl;
    exit(-1);
  }
  // default rwkv eos is "\0"
  std::cout << "Done!\n" << std::endl;
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