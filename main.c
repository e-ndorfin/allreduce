#include "protocol.h"
#include "worker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define DEFAULT_NUM_WORKERS 3
#define DEFAULT_VECTOR_LEN 6

int main(int argc, char *argv[])
{
    int N = DEFAULT_NUM_WORKERS;
    int L = DEFAULT_VECTOR_LEN;

    // parse -n and -l from argv
    if (argc > 1)
    {
        for (int argi = 1; argi < argc; argi++)
        {
            if (strcmp(argv[argi], "-n") == 0)
            {
                N = atoi(argv[argi + 1]);
                argi++;
            }
            if (strcmp(argv[argi], "-l") == 0)
            {
                L = atoi(argv[argi + 1]);
                argi++;
            }
        }
    }
    // validate number of workers and vector size is adequate
    if (!(N >= 3 && L >= N))
    {
        fprintf(stderr, "Error: need at least 3 workers (got %d vectors, length %d)\n", N, L);
        exit(1);
    }

    setup_sigpipe_handler();

    printf("Ring AllReduce: %d workers, vector length %d\n", N, L);

    /* ==========================================================
     *  2. GENERATE INPUT VECTORS
     * ==========================================================
     *  For now im just gonna flag this (currently used for testing), but Arnav,
     *  you will definitely need to change this for when you implement the actual
     *  backprop steps (input vectors wont be fixed...)
     * ========================================================== */

    float **inputs = malloc(sizeof(float *) * N);
    for (int i = 0; i < N; i++)
    {
        inputs[i] = malloc(sizeof(float) * L);
        for (int j = 0; j < L; j++)
            inputs[i][j] = (float)(i + 1);
    }

    printf("Input Vectors:\n");
    for (int i = 0; i < N; i++)
    {
        printf("  Worker %d: [", i);
        for (int j = 0; j < L; j++)
            printf("%.1f%s", inputs[i][j], j < L - 1 ? ", " : "");
        printf("]\n");
    }
    printf("\n");

    // allocate pipes: init (parent->worker), ring (worker<->worker), result (worker 0->parent)
    int (*init_pipe)[2] = malloc(sizeof(int[2]) * N);
    int (*ring_pipe)[2] = malloc(sizeof(int[2]) * N);
    int result_pipe[2];

    for (int i = 0; i < N; i++)
    {
        if (pipe(init_pipe[i]) == -1)
        {
            perror("init_pipe");
            exit(1);
        }
        if (pipe(ring_pipe[i]) == -1)
        {
            perror("ring_pipe");
            exit(1);
        }
    }
    if (pipe(result_pipe) == -1)
    {
        perror("result_pipe");
        exit(1);
    }

    // allocate pid's
    pid_t *pids = malloc(sizeof(pid_t) * N);

    for (int i = 0; i < N; i++)
    {
        pid_t pid = fork();
        if (pid == -1)
        {
            perror("fork");
            exit(1);
        }
        if (pid == 0)
        {
            // child
            int init_read_fd = init_pipe[i][0];
            int ring_read_fd = ring_pipe[(i - 1 + N) % N][0]; // previous node
            int ring_write_fd = ring_pipe[i][1];              // current node
            int result_fd = (i == 0) ? result_pipe[1] : -1;

            // close fds this child doesn't own
            for (int j = 0; j < N; j++)
            {
                close(init_pipe[j][1]); // close all init write ends
                if (j != i)
                    close(init_pipe[j][0]); // close other workers' init read ends
                if (ring_pipe[j][0] != ring_read_fd)
                    close(ring_pipe[j][0]);
                if (ring_pipe[j][1] != ring_write_fd)
                    close(ring_pipe[j][1]);
            }
            close(result_pipe[0]);
            if (i != 0)
            {
                close(result_pipe[1]);
            }

            worker_main(init_read_fd, ring_read_fd, ring_write_fd, result_fd);
        }
        pids[i] = pid;
    }

    // Parent: send INIT to each worker over init pipes
    for (int i = 0; i < N; i++)
    {
        close(init_pipe[i][0]); // close read ends
        if (send_init(init_pipe[i][1], i, N, L, inputs[i]) != 0)
        {
            fprintf(stderr, "Error: failed to send INIT to worker %d\n", i);
            exit(1);
        }
        close(init_pipe[i][1]); // close write end after sending
    }

    // Close ring and result pipe ends the parent doesn't use
    for (int i = 0; i < N; i++)
    {
        close(ring_pipe[i][0]);
        close(ring_pipe[i][1]);
    }
    close(result_pipe[1]);

    done_hdr_t done_hdr;
    float *result_data = NULL;

    if (recv_done(result_pipe[0], &done_hdr, &result_data) != 0)
    {
        fprintf(stderr, "Error: failed to receive DONE from worker 0\n");
        exit(1);
    }
    close(result_pipe[0]);

    printf("Final Result:\n[");
    for (int j = 0; j < done_hdr.vec_len; j++)
        printf("%.1f%s", result_data[j], j < done_hdr.vec_len - 1 ? ", " : "");
    printf("]\n\n");

    // NOTE: verification below is for testing only — remove for final project
    float expected = (float)(N * (N + 1)) / 2.0f;
    int pass = 1;
    for (int j = 0; j < done_hdr.vec_len; j++)
    {
        if (result_data[j] != expected)
        {
            fprintf(stderr, "FAIL: result[%d] = %.2f, expected %.2f\n",
                    j, result_data[j], expected);
            pass = 0;
        }
    }
    printf("Verification: %s\n", pass ? "PASS" : "FAIL");
    free(result_data);

    // waiting on children, error handling
    for (int i = 0; i < N; i++)
    {
        int status;
        waitpid(pids[i], &status, 0);
        if (WIFEXITED(status) && WEXITSTATUS(status) != 0)
        {
            fprintf(stderr, "Warning: worker %d exited with status %d\n",
                    i, WEXITSTATUS(status));
        }
        else if (WIFSIGNALED(status))
        {
            fprintf(stderr, "Warning: worker %d killed by signal %d\n",
                    i, WTERMSIG(status));
        }
    }

    // final cleanups
    printf("Overall: %s\n", pass ? "PASS" : "FAIL");

    for (int i = 0; i < N; i++)
        free(inputs[i]);
    free(inputs);
    free(init_pipe);
    free(ring_pipe);
    free(pids);

    return pass ? 0 : 1;
}
