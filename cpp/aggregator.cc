#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

REGISTER_OP("Aggregator")
    .Input("value: float32")
    .Input("advantage: float32")
    .Output("Q: float32");

class AggregatorOp : public OpKernel {
public:
    explicit AggregatorOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
    // TODO: make Aggregate op
    }
};

REGISTER_KERNEL_BUILDER(Name("Aggregator").Device(DEVICE_CPU), AggregatorOp);