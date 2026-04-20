// NOTE: The implementations for forward and backward pass live in gnn_*.c

#include "layers.h"
// Reset gradient
void sage_layer_zero_gradients(SageLayer* l)
{
    real_zero_out(l->grad_input, l->num_nodes * l->in_dim);
    real_zero_out(l->grad_Wagg, l->in_dim * l->out_dim);
    real_zero_out(l->grad_Wroot, l->in_dim * l->out_dim);
}

void relu_layer_zero_gradients(ReluLayer* l)
{
    real_zero_out(l->grad_input, l->num_nodes * l->dim);
}

void normalize_layer_zero_gradients(L2NormLayer* l)
{
    real_zero_out(l->grad_input, l->num_nodes * l->dim);
}

void linear_layer_zero_gradients(LinearLayer* l)
{
    real_zero_out(l->grad_input, l->num_nodes * l->in_dim);
    real_zero_out(l->grad_W, l->in_dim * l->out_dim);
    real_zero_out(l->grad_bias, l->out_dim);
}

void logsoft_layer_zero_gradients(LogSoftmaxLayer* l)
{
    real_zero_out(l->grad_input, l->num_nodes * l->dim);
}
