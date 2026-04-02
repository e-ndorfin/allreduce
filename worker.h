#ifndef WORKER_H
#define WORKER_H

/*
 * worker_main — entry point for each forked child process.
 *
 * Called from main.c after fork(). This function should NOT return;
 * call _exit() when done.
 *
 * Parameters:
 *   init_read_fd   — read end of the init pipe from parent.
 *                    worker calls recv_init() on this fd to obtain its
 *                    worker ID, num_workers, vec_len, and gradient vector.
 *   ring_read_fd   — read end of pipe from LEFT neighbor (worker (id-1+N)%N)
 *   ring_write_fd  — write end of pipe to RIGHT neighbor (worker (id+1)%N)
 *   result_fd      — write end of result pipe back to parent.
 *                    only valid for worker 0 (result_fd >= 0).
 *                    for all other workers this is -1, ignore it.
 */

void worker_main(int init_read_fd,
                 int ring_read_fd, int ring_write_fd, int result_fd);

#endif
