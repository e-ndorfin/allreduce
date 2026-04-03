#include "worker.h"
#include "protocol.h"
#include "Tinn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int chunk_len(int chunk_idx, int chunk_size, int num_workers, int vec_len)
{
    if (chunk_idx == num_workers - 1)
        return vec_len - chunk_idx * chunk_size;
    return chunk_size;
}

// Runs ring allreduce on vec (in-place). vec has vec_len floats, N workers.
static void ring_allreduce(float *vec, int vec_len, int id, int N,
                           int chunk_size, float *recv_buf,
                           int ring_read_fd, int ring_write_fd)
{
    // Phase 1: scatter-reduce
    for (int r = 0; r < N - 1; r++)
    {
        int send_idx = (id - r + N) % N;
        int recv_idx = (id - r - 1 + N) % N;

        int send_clen = chunk_len(send_idx, chunk_size, N, vec_len);
        int recv_clen = chunk_len(recv_idx, chunk_size, N, vec_len);

        float *send_ptr = vec + send_idx * chunk_size;
        float *recv_ptr = vec + recv_idx * chunk_size;

        send_chunk(ring_write_fd, PHASE_SCATTER_REDUCE, send_idx, send_clen, send_ptr);

        chunk_hdr_t hdr;
        recv_chunk(ring_read_fd, &hdr, recv_buf);

        for (int j = 0; j < recv_clen; j++)
            recv_ptr[j] += recv_buf[j];
    }

    // Phase 2: allgather
    for (int r = 0; r < N - 1; r++)
    {
        int send_idx = (id - r + 1 + N) % N;
        int recv_idx = (id - r + N) % N;

        int send_clen = chunk_len(send_idx, chunk_size, N, vec_len);
        int recv_clen = chunk_len(recv_idx, chunk_size, N, vec_len);

        float *send_ptr = vec + send_idx * chunk_size;
        float *recv_ptr = vec + recv_idx * chunk_size;

        send_chunk(ring_write_fd, PHASE_ALLGATHER, send_idx, send_clen, send_ptr);

        chunk_hdr_t hdr;
        recv_chunk(ring_read_fd, &hdr, recv_buf);

        memcpy(recv_ptr, recv_buf, recv_clen * sizeof(float));
    }
}

void worker_main(int init_read_fd,
                 int ring_read_fd, int ring_write_fd, int result_fd)
{
    // Receive INIT from parent
    init_hdr_t init_hdr;
    float *weights = NULL;
    float *dataset = NULL;
    if (recv_init(init_read_fd, &init_hdr, &weights, &dataset) != 0)
    {
        _exit(1);
    }
    close(init_read_fd);

    int id = init_hdr.worker_id;
    int N = init_hdr.num_workers;         // number of workers
    int nips = init_hdr.nips;             // number of inputs
    int nhid = init_hdr.nhid;             // number of hidden neurons
    int nops = init_hdr.nops;             // number of outputs
    int num_rows = init_hdr.num_rows;     // number of rows in data
    int num_epochs = init_hdr.num_epochs; // num epochs
    float rate = init_hdr.learning_rate;  // learning rate
    float anneal = init_hdr.anneal;       // annealing factor
    int param_count = init_hdr.vec_len;   // gradient vector size
    int cols = nips + nops;

    // Build network and load initial weights
    Tinn tinn = xtbuild(nips, nhid, nops);
    xtsetweights(tinn, weights);
    free(weights);

    // Compute which shard this worker uses
    int rows_per_worker = num_rows / N;
    int shard_start = id * rows_per_worker;
    int shard_end = (id == N - 1) ? num_rows : (id + 1) * rows_per_worker;
    int shard_rows = shard_end - shard_start;

    int chunk_size = param_count / N;

    float *epoch_grad = malloc(sizeof(float) * param_count);
    float *sample_grad = malloc(sizeof(float) * param_count);
    float *recv_buf = malloc(sizeof(float) * (chunk_size + N));

    // Training loop (some of this was taken from tinn/test.c training loop code)
    // https://github.com/glouw/tinn/blob/master/test.c

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
        // Zero accumulated gradients
        memset(epoch_grad, 0, sizeof(float) * param_count);
        float epoch_error = 0.0f;

        // Forward + backward on this worker's shard
        for (int s = shard_start; s < shard_end; s++)
        {
            const float *in = dataset + s * cols;
            const float *tg = dataset + s * cols + nips;

            xtforward(tinn, in);
            epoch_error += xtbackward(tinn, in, tg, rate, sample_grad);

            for (int k = 0; k < param_count; k++)
                epoch_grad[k] += sample_grad[k];
        }

        // Ring allreduce to sum gradients across all workers
        ring_allreduce(epoch_grad, param_count, id, N,
                       chunk_size, recv_buf,
                       ring_read_fd, ring_write_fd);

        // Average over total number of samples
        for (int k = 0; k < param_count; k++)
            epoch_grad[k] /= num_rows;

        // Apply averaged gradients to weights
        xtapply(tinn, epoch_grad);

        // Anneal learning rate
        rate *= anneal;

        if (id == 0)
        {
            printf("Epoch %3d: error %.6f, rate %.4f\n",
                   epoch, (double)epoch_error / shard_rows, (double)rate);
            fflush(stdout);
        }
    }

    // Worker 0 sends final weights back to parent
    if (id == 0)
    {
        float *final_weights = malloc(sizeof(float) * param_count);
        xtgetweights(tinn, final_weights);
        send_done(result_fd, id, param_count, final_weights);
        free(final_weights);
    }

    // Cleanup
    close(ring_read_fd);
    close(ring_write_fd);
    if (result_fd >= 0)
        close(result_fd);

    free(epoch_grad);
    free(sample_grad);
    free(recv_buf);
    free(dataset);
    xtfree(tinn);
    _exit(0);
}
