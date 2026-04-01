#include "worker.h"
#include "protocol.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int chunk_len(int chunk_idx, int chunk_size, int num_workers, int vec_len)
{
    if (chunk_idx == num_workers - 1)
        return vec_len - chunk_idx * chunk_size;
    return chunk_size;
}

void worker_main(int id, int num_workers, int vec_len, float *input,
                 int ring_read_fd, int ring_write_fd, int result_fd)
{
    int N = num_workers;
    int chunk_size = vec_len / N;

    // Recv buffer — sized for the largest possible chunk (last chunk with remainder)
    float *recv_buf = malloc(sizeof(float) * (chunk_size + N));

    // First step: scatter-reduce. Send chunk[(i-r+N)%N], recv chunk[(i-r-1+N)%N], then aggregate
    for (int r = 0; r < N - 1; r++) {
        int send_idx = (id - r + N) % N;
        int recv_idx = (id - r - 1 + N) % N;

        int send_clen = chunk_len(send_idx, chunk_size, N, vec_len);
        int recv_clen = chunk_len(recv_idx, chunk_size, N, vec_len);

        float *send_ptr = input + send_idx * chunk_size;
        float *recv_ptr = input + recv_idx * chunk_size;

        send_chunk(ring_write_fd, PHASE_SCATTER_REDUCE, send_idx, send_clen, send_ptr);

        chunk_hdr_t hdr;
        recv_chunk(ring_read_fd, &hdr, recv_buf);

        for (int j = 0; j < recv_clen; j++)
            recv_ptr[j] += recv_buf[j];
    }

    // Second step: all gather. send chunk[(i-r+1+N)%N], recv chunk[(i-r+N)%N], memcpy 
    for (int r = 0; r < N - 1; r++) {
        int send_idx = (id - r + 1 + N) % N;
        int recv_idx = (id - r + N) % N;

        int send_clen = chunk_len(send_idx, chunk_size, N, vec_len);
        int recv_clen = chunk_len(recv_idx, chunk_size, N, vec_len);

        float *send_ptr = input + send_idx * chunk_size;
        float *recv_ptr = input + recv_idx * chunk_size;

        send_chunk(ring_write_fd, PHASE_ALLGATHER, send_idx, send_clen, send_ptr);

        chunk_hdr_t hdr;
        recv_chunk(ring_read_fd, &hdr, recv_buf);

        memcpy(recv_ptr, recv_buf, recv_clen * sizeof(float));
    }

    // Send back to parent
    if (id == 0)
        send_done(result_fd, id, vec_len, input);

    close(ring_read_fd);
    close(ring_write_fd);
    if (result_fd >= 0)
        close(result_fd);

    free(recv_buf);
    free(input);
    _exit(0);
}
