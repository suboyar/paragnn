void relu(ReluLayer *const l)
{
    TIMER_FUNC();

#pragma omp parallel for
    for (size_t i = 0; i < l->num_nodes; i++) {
#pragma omp simd
        for (size_t j = 0; j < l->dim; j++) {
            l->output[i*l->dim+j] = fmax(0.0, l->input[i*l->dim+j]);
        }
    }
}

void relu_backward(ReluLayer *const l)
{
    TIMER_FUNC();

    size_t n = l->num_nodes * l->dim;

    // TODO: grad_input = grad_output * fmaxf(0.0f, copysignf(1.0f, output));
#pragma omp parallel for
    for (size_t i = 0; i < n; i++)
    {
        l->grad_input[i] = (l->output[i] > 0.0) ? l->grad_output[i] : 0.0;
    }
}

void logsoft(LogSoftLayer *const l)
{
    TIMER_FUNC();

#pragma omp parallel for
    for (size_t i = 0; i < l->num_nodes; i++)
    {
        double max = l->input[i*l->dim];

        for (size_t j = 1; j < l->num_classes; j++)
        {
            max = fmax(max, l->input[i*l->dim+j]);
        }

        double logsumexp = 0.0;
        for (size_t j = 0; j < l->num_classes; j++)
        {
            logsumexp += exp(l->input[i*l->dim+j] - max);
        }

        logsumexp = log(logsumexp);

        for (size_t j = 0; j < l->num_classes; j++)
        {
            l->output[i*l->dim+j] = l->input[i*l->dim+j] - max - logsumexp;
        }
    }

    nob_log(NOB_INFO, "log_softmax: ok");
}

void cross_entropy_backward(LogSoftLayer *const l, uint32_t *labels)
{
    TIMER_FUNC();

    uint32_t num_classes = l->num_classes;

    double scale = 1.0 / l->num_nodes;
#pragma omp parallel for
    for (size_t i = 0; i < l->num_nodes; i++) {
        uint32_t target = labels[i];
        for (size_t j = 0; j < num_classes; j++) {
            double softmax_val = exp(l->output[i*l->dim+j]);
            if (j == target)
                l->grad_input[i*l->dim+j] = (softmax_val - 1) * scale;
            else
                l->grad_input[i*l->dim+j] = softmax_val * scale;
        }
    }

    nob_log(NOB_INFO, "cross_entropy_backward: ok");
}

float nll_loss(LogSoftLayer *l, const uint32_t *labels)
{
    TIMER_FUNC();

    float loss = 0.0;
#pragma omp parallel for reduction(+:loss)
    for (size_t i = 0; i < l->num_nodes; i++)
    {
        loss -= (float)l->output[i * l->num_classes + labels[i]];
    }

    return loss / l->num_nodes;
}

float accuracy(const LogSoftLayer *l, const uint32_t *labels)
{
    TIMER_FUNC();

    uint64_t correct = 0;
#pragma omp parallel for reduction(+:correct)
    for (size_t i = 0; i < l->num_nodes; i++)
    {
        const Real *row = &l->output[i * l->num_classes];
        uint32_t pred_class = 0;
        for (size_t j = 1; j < l->num_classes; j++)
        {
            if (row[j] > row[pred_class])
                pred_class = j;
        }

        if (pred_class == labels[i])
            correct++;
    }

    return (float)correct / l->num_nodes;
}
