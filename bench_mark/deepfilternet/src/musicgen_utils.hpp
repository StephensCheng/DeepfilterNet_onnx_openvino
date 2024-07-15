#ifndef MUSICGEN_UTILS_HPP
#define MUSICGEN_UTILS_HPP

#include <iostream>
#include <torch/torch.h>
#include <openvino/openvino.hpp>

static torch::Tensor wrap_ov_tensor_as_torch(ov::Tensor ov_tensor)
{
    if (!ov_tensor)
    {
        throw std::invalid_argument("wrap_ov_tensor_as_torch: invalid ov_tensor");
    }

    // first, determine torch dtype from ov type
    at::ScalarType torch_dtype;
    size_t element_byte_size;
    void *pOV_Tensor;
    switch (ov_tensor.get_element_type())
    {
    case ov::element::i8:
        torch_dtype = torch::kI8;
        element_byte_size = sizeof(unsigned char);
        pOV_Tensor = ov_tensor.data();
        break;

    case ov::element::f32:
        torch_dtype = torch::kFloat32;
        element_byte_size = sizeof(float);
        pOV_Tensor = ov_tensor.data<float>();
        break;

    case ov::element::f16:
        torch_dtype = torch::kFloat16;
        element_byte_size = sizeof(short);
        pOV_Tensor = ov_tensor.data<ov::float16>();
        break;

    case ov::element::i64:
        torch_dtype = torch::kInt64;
        element_byte_size = sizeof(int64_t);
        pOV_Tensor = ov_tensor.data<int64_t>();
        break;

    default:
        std::cout << "type = " << ov_tensor.get_element_type() << std::endl;
        throw std::invalid_argument("wrap_ov_tensor_as_torch: unsupported type");
        break;
    }

    // fill torch shape
    std::vector<int64_t> torch_shape;
    for (auto s : ov_tensor.get_shape())
        torch_shape.push_back(s);
    std::vector<int64_t> torch_strides;
    for (auto s : ov_tensor.get_strides())
        torch_strides.push_back(s / element_byte_size); //<- torch stride is in term of # of elements, whereas openvino stride is in terms of bytes

    auto options =
        torch::TensorOptions()
            .dtype(torch_dtype);

    return torch::from_blob(pOV_Tensor, torch_shape, torch_strides, options);
}

static inline std::string FullPath(std::string base_dir, std::string filename)
{
    const std::string os_sep = "/";
    return base_dir + os_sep + filename;
}

class MultiFrameModule
{
public:
    MultiFrameModule(int64_t num_freqs, int64_t frame_size, int64_t lookahead = 0, bool real = false);

    virtual torch::Tensor forward(torch::Tensor spec, torch::Tensor coefs) = 0;

protected:
    torch::Tensor spec_unfold(torch::Tensor spec);

    int64_t _num_freqs;
    int64_t _frame_size;
    bool _real;
    bool _need_unfold;
    int64_t _lookahead;

    torch::nn::ConstantPad3d _pad3d{nullptr}; // For 3D padding
    torch::nn::ConstantPad2d _pad2d{nullptr}; // For 2D padding
};

class DF : public MultiFrameModule
{
public:
    DF(int64_t num_freqs, int64_t frame_size, int64_t lookahead = 0, bool conj = false);

    virtual torch::Tensor forward(torch::Tensor spec, torch::Tensor coefs) override;

private:
    bool _conj;
};

#endif