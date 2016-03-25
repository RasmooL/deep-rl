#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

REGISTER_OP("Gate")
    .Input("prev: float32")
    .Input("cur: float32")
    .Output("pred: float32");

class GateOp : public OpKernel {
public:
    explicit GateOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
	const Tensor& prev_tensor = context->input(0);
	const Tensor& cur_tensor = context->input(1);

	// allocate output tensor
    Tensor* pred_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, cur_tensor.shape(), &pred_tensor));

    // TODO: finish Gate op
    }
};

REGISTER_KERNEL_BUILDER(Name("Gate").Device(DEVICE_CPU), GateOp);