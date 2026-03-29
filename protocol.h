#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stddef.h>
#include <unistd.h>

// Types of messages
typedef enum {
    MSG_INIT,
    MSG_CHUNK,
    MSG_DONE,
    MSG_SHUTDOWN
} msg_type_t;

// Phases of allreduce operation
typedef enum {
    PHASE_SCATTER_REDUCE,
    PHASE_ALLGATHER
} phase_t;

// Message headers

typedef struct {
    msg_type_t type;
    int worker_id;
    int num_workers;
    int vec_len;
} init_hdr_t;

typedef struct {
    msg_type_t type;
    phase_t phase;
    int chunk_idx;
    int chunk_len;
} chunk_hdr_t;

typedef struct {
    msg_type_t type;
    int worker_id;
    int vec_len;
} done_hdr_t;

typedef struct {
    msg_type_t type;
} shutdown_hdr_t;

// I/O helpers

ssize_t read_all(int fd, void *buf, size_t count);
ssize_t write_all(int fd, const void *buf, size_t count);

// Message send / recv

int send_init(int fd, int worker_id, int num_workers,
              int vec_len, const float *data);
int recv_init(int fd, init_hdr_t *hdr, float **data);

int send_chunk(int fd, phase_t phase, int chunk_idx,
               int chunk_len, const float *data);
int recv_chunk(int fd, chunk_hdr_t *hdr, float *buf);

int send_done(int fd, int worker_id, int vec_len, const float *data);
int recv_done(int fd, done_hdr_t *hdr, float **data);

int send_shutdown(int fd);

// Signal handling

void setup_sigpipe_handler(void);

#endif
