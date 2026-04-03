#include "protocol.h"
#include "worker.h"
#include "Tinn.h"
#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <time.h>

#define DEFAULT_NUM_WORKERS 4
#define NIPS 256
#define NHID 28
#define NOPS 10
#define NUM_EPOCHS 256
#define LEARNING_RATE 5.0f
#define ANNEAL 0.99f

int main(int argc, char *argv[])
{
    int N = DEFAULT_NUM_WORKERS;

    int nhid = NHID;
    int num_epochs = NUM_EPOCHS;

    // Input arguments
    for (int argi = 1; argi < argc; argi++)
    {
        if (strcmp(argv[argi], "--n") == 0 && argi + 1 < argc)
        {
            N = atoi(argv[argi + 1]);
            argi++;
        }
        else if (strcmp(argv[argi], "--hidden_num") == 0 && argi + 1 < argc)
        {
            nhid = atoi(argv[argi + 1]);
            argi++;
        }
        else if (strcmp(argv[argi], "--epoch_num") == 0 && argi + 1 < argc)
        {
            num_epochs = atoi(argv[argi + 1]);
            argi++;
        }
        else
        {
            fprintf(stderr, "Usage: %s [--n NUM_WORKERS] [--hidden_num HIDDEN] [--epoch_num EPOCHS]\n", argv[0]);
            fprintf(stderr, "Unknown argument: %s\n", argv[argi]);
            exit(1);
        }
    }

    if (N < 2)
    {
        fprintf(stderr, "Error: need at least 2 workers (got %d)\n", N);
        exit(1);
    }
    if (nhid < 1)
    {
        fprintf(stderr, "Error: need at least 1 hidden neuron (got %d)\n", nhid);
        exit(1);
    }
    if (num_epochs < 1)
    {
        fprintf(stderr, "Error: need at least 1 epoch (got %d)\n", num_epochs);
        exit(1);
    }

    setup_sigpipe_handler();
    srand(time(0));

    printf("Distributed Training: %d workers, %d epochs, %d hidden\n", N, num_epochs, nhid);

    // Load training data
    Data data = data_load("semeion.data", NIPS, NOPS);
    printf("Loaded %d samples (%d inputs, %d outputs)\n", data.rows, data.nips, data.nops);

    // Build network and extract initial weights
    Tinn tinn = xtbuild(NIPS, nhid, NOPS);
    int param_count = tinn.nw + tinn.nb;
    float *init_weights = malloc(sizeof(float) * param_count);
    xtgetweights(tinn, init_weights);

    printf("Network: %d -> %d -> %d (%d parameters)\n", NIPS, nhid, NOPS, param_count);

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
                close(init_pipe[j][1]); // init pipe READ ONLY for children
                if (j != i)
                    close(init_pipe[j][0]); // close other workers' init read ends

                // Close all irrelevant pipe read/write ends
                if (ring_pipe[j][0] != ring_read_fd)
                    close(ring_pipe[j][0]);
                if (ring_pipe[j][1] != ring_write_fd)
                    close(ring_pipe[j][1]);
            }

            // Close all children read-ends of result_pipe
            close(result_pipe[0]);

            if (i != 0)
            {
                close(result_pipe[1]);
            }

            // Free parent-only resources before entering worker
            free(init_weights);
            free(pids);
            xtfree(tinn);

            worker_main(init_read_fd, ring_read_fd, ring_write_fd, result_fd);
            // worker_main calls _exit, should not reach here
        }
        pids[i] = pid;
    }

    // Parent: send INIT to each worker over init pipes
    for (int i = 0; i < N; i++)
    {
        close(init_pipe[i][0]); // close read ends

        init_hdr_t hdr;
        hdr.type = MSG_INIT;
        hdr.worker_id = i;
        hdr.num_workers = N;
        hdr.nips = NIPS;
        hdr.nhid = nhid;
        hdr.nops = NOPS;
        hdr.num_rows = data.rows;
        hdr.num_epochs = num_epochs;
        hdr.learning_rate = LEARNING_RATE;
        hdr.anneal = ANNEAL;
        hdr.vec_len = param_count;
        hdr.data_len = data.rows * (NIPS + NOPS);

        if (send_init(init_pipe[i][1], &hdr, init_weights, data.flat) != 0)
        {
            fprintf(stderr, "Error: failed to send INIT to worker %d\n", i);
            exit(1);
        }
        close(init_pipe[i][1]); // close write end after sending
    }

    // Close parent's ring and result pipe ends
    for (int i = 0; i < N; i++)
    {
        close(ring_pipe[i][0]);
        close(ring_pipe[i][1]);
    }
    close(result_pipe[1]); // read-only for parent

    // Receive final model from worker 0
    done_hdr_t done_hdr;
    float *final_weights = NULL;

    if (recv_done(result_pipe[0], &done_hdr, &final_weights) != 0)
    {
        fprintf(stderr, "Error: failed to receive DONE from worker 0\n");
        exit(1);
    }
    close(result_pipe[0]);

    // Load trained weights into network
    xtsetweights(tinn, final_weights);

    // Verify accuracy on the dataset
    int correct = 0;
    for (int s = 0; s < data.rows; s++)
    {
        const float *in = data.flat + s * (NIPS + NOPS);
        const float *tg = data.flat + s * (NIPS + NOPS) + NIPS;
        const float *pd = xtpredict(tinn, in);

        // Find argmax of prediction and target
        int pred_max = 0, targ_max = 0;
        for (int j = 1; j < NOPS; j++)
        {
            if (pd[j] > pd[pred_max])
                pred_max = j;
            if (tg[j] > tg[targ_max])
                targ_max = j;
        }
        if (pred_max == targ_max)
            correct++;
    }
    printf("\nFinal Accuracy: %d/%d (%.1f%%)\n",
           correct, data.rows, 100.0 * correct / data.rows);

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
    free(final_weights);
    free(init_weights);
    free(init_pipe);
    free(ring_pipe);
    free(pids);
    xtfree(tinn);
    data_free(data);

    return 0;
}
