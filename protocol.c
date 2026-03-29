#include "protocol.h"
#include <stdio.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>

// I/O helpers

ssize_t read_all(int fd, void *buf, size_t count) {
    ssize_t recv_bytes = 0;
    while (recv_bytes < (ssize_t)count) {
        // Read up until where we left off in the buffer
        ssize_t ret = read(fd, (char *)buf + recv_bytes, count - recv_bytes);
        
        // Handle error when reading from buffer
        if (ret < 0) {  
            if (errno == EINTR) {
                continue; // Retry if interrupted by signal
            }
            perror("read");
            return -1;
        }

        recv_bytes += ret;

        if (ret == 0) {  // Writer closed pipe
            return recv_bytes;
        }
    }
    return recv_bytes;

}

ssize_t write_all(int fd, const void *buf, size_t count) {
    ssize_t written_bytes = 0;
    while (written_bytes < (ssize_t)count) {
        // Write up until where we left off in the buffer
        ssize_t ret = write(fd, (char *)buf + written_bytes, count - written_bytes);
        
        // Handle error when writing to buffer
        if (ret < 0) {  
            if (errno == EINTR) {
                continue; // Retry if interrupted by signal
            }
            perror("write");
            return -1;
        }

        written_bytes += ret;
    }
    return written_bytes;
}

// Message send / receive

// Parent sending INIT message + data to worker
int send_init(int fd, int worker_id, int num_workers,
              int vec_len, const float *data) {
    
    init_hdr_t header;
    header.type = MSG_INIT;
    header.worker_id = worker_id;
    header.num_workers = num_workers;
    header.vec_len = vec_len;

    ssize_t header_size = write_all(fd, &header, sizeof(init_hdr_t));
    ssize_t data_size = write_all(fd, data, sizeof(float) * vec_len);

    if (header_size == -1 || data_size == -1) {
        return -1;
    }

    return 0;
}

// Worker receiving INIT message + data
int recv_init(int fd, init_hdr_t *hdr, float **data) {

    ssize_t header_size = read_all(fd, hdr, sizeof(init_hdr_t));

    if (header_size == -1) {
        return -1;
    }
    
    *data = malloc(sizeof(float) * hdr->vec_len);
    ssize_t data_size = read_all(fd, *data, sizeof(float) * hdr->vec_len);

    if (data_size == -1) {
        return -1;
    }

    return 0;
}

// Worker -> Worker (used in both scatter-reduce and allgather)
int send_chunk(int fd, phase_t phase, int chunk_idx,
               int chunk_len, const float *data) {

    chunk_hdr_t header;
    header.type = MSG_CHUNK;
    header.phase = phase;
    header.chunk_idx = chunk_idx;
    header.chunk_len = chunk_len;

    ssize_t header_size = write_all(fd, &header, sizeof(chunk_hdr_t));
    ssize_t data_size = write_all(fd, data, sizeof(float) * chunk_len);

    if (header_size == -1 || data_size == -1) {
        return -1;
    }

    return 0;
}

// Worker -> Worker (receiving chunk)
int recv_chunk(int fd, chunk_hdr_t *hdr, float *buf) {

    ssize_t header_size = read_all(fd, hdr, sizeof(chunk_hdr_t));

    if (header_size == -1) {
        return -1;
    }

    ssize_t data_size = read_all(fd, buf, sizeof(float) * hdr->chunk_len);

    if (data_size == -1) {
        return -1;
    }

    return 0;
}

// Worker -> Parent (sending final result)
int send_done(int fd, int worker_id, int vec_len, const float *data) {

    done_hdr_t header;
    header.type = MSG_DONE;
    header.worker_id = worker_id;
    header.vec_len = vec_len;

    ssize_t header_size = write_all(fd, &header, sizeof(done_hdr_t));
    ssize_t data_size = write_all(fd, data, sizeof(float) * vec_len);

    if (header_size == -1 || data_size == -1) {
        return -1;
    }

    return 0;
}

// Parent receiving final result from worker
int recv_done(int fd, done_hdr_t *hdr, float **data) {

    ssize_t header_size = read_all(fd, hdr, sizeof(done_hdr_t));

    if (header_size == -1) {
        return -1;
    }

    *data = malloc(sizeof(float) * hdr->vec_len);
    ssize_t data_size = read_all(fd, *data, sizeof(float) * hdr->vec_len);

    if (data_size == -1) {
        free(*data);
        return -1;
    }

    return 0;
}

// Parent -> Worker (error cleanup)
int send_shutdown(int fd) {

    shutdown_hdr_t header;
    header.type = MSG_SHUTDOWN;

    ssize_t header_size = write_all(fd, &header, sizeof(shutdown_hdr_t));

    if (header_size == -1) {
        return -1;
    }

    return 0;
}

// Signal handling

void setup_sigpipe_handler(void) {
    signal(SIGPIPE, SIG_IGN);
}
