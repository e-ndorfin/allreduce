#ifndef WORKER_H
#define WORKER_H

/*
 * worker_main — entry point for each forked child process.
 *
 * Called from main.c after fork(). This function should NOT return;
 * call _exit() when done.
 *
 * Parameters:
 *   id             — this worker's index in the ring (0 to N-1)
 *   num_workers    — total number of workers (N)
 *   vec_len        — length of the gradient vector (L). guaranteed L >= N.
 *   input          — this worker's local gradient vector (float[vec_len]).
 *                    you own this memory, free it before exiting.
 *   ring_read_fd   — read end of pipe from LEFT neighbor (worker (id-1+N)%N)
 *   ring_write_fd  — write end of pipe to RIGHT neighbor (worker (id+1)%N)
 *   result_fd      — write end of result pipe back to parent.
 *                    only valid for worker 0 (result_fd >= 0).
 *                    for all other workers this is -1, ignore it.
 *
 * What you need to implement inside worker_main:
 *
 *   1. SCATTER-REDUCE  (N-1 rounds)
 *      - Chunk size = vec_len / num_workers  (integer division)
 *      - In round r, worker i:
 *          • sends chunk[(i - r + N) % N] to the RIGHT  (send_chunk, PHASE_SCATTER_REDUCE)
 *          • recvs chunk[(i - r - 1 + N) % N] from the LEFT  (recv_chunk)
 *          • accumulates (+=) the received data into your local vector
 *      - After N-1 rounds, each worker owns one fully-reduced chunk.
 *
 *   2. ALLGATHER  (N-1 rounds)
 *      - Same ring direction, but now you REPLACE instead of accumulate.
 *      - In round r, worker i:
 *          • sends chunk[(i - r + 1 + N) % N] to the RIGHT  (send_chunk, PHASE_ALLGATHER)
 *          • recvs chunk[(i - r + N) % N] from the LEFT  (recv_chunk)
 *          • copies received data into your local vector (memcpy, not +=)
 *      - After N-1 rounds, every worker has the complete reduced vector.
 *
 *   3. DONE
 *      - If id == 0: send the final vector to parent via send_done(result_fd, ...)
 *      - Close ring_read_fd, ring_write_fd, and result_fd (if valid).
 *      - Free input and any other allocations.
 *      - _exit(0)
 *
 * Available protocol helpers (see protocol.h):
 *   send_chunk(fd, phase, chunk_idx, chunk_len, data)
 *   recv_chunk(fd, &hdr, buf)
 *   send_done(fd, worker_id, vec_len, data)
 *
 * Chunk indexing helper:
 *   chunk k starts at  input[k * chunk_size]
 *   chunk k has length  chunk_size  (for k < N-1)
 *                    or vec_len - k * chunk_size  (for k == N-1, handles remainder)
 */

void worker_main(int id, int num_workers, int vec_len, float *input,
                 int ring_read_fd, int ring_write_fd, int result_fd);

#endif
