/*
Aggregator for A3C.
NOTE: Not being worked on.

Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. See the LICENSE.txt file for details.
*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

REGISTER_OP("Aggregator")
    .Input("value: float32")
    .Input("advantage: float32")
    .Output("q: float32");

class AggregatorOp : public OpKernel {
public:
    explicit AggregatorOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& value_tensor = context->input(0);
        const Tensor& advantage_tensor = context->input(1);

	    // allocate output tensor
        Tensor* Q_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, advantage_tensor.shape(), &Q_tensor));

        // TODO: finish Aggregator op
    }
};

REGISTER_KERNEL_BUILDER(Name("Aggregator").Device(DEVICE_CPU), AggregatorOp);
