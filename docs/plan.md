# Ring Allreduce — Project Plan

## The Problem

You have N processes, each holding its own vector of floats (think of them as gradient vectors from a machine learning training step). The goal: every process should end up with the element-wise sum of all N vectors, without any single process ever collecting everyone's full data. This is the core operation behind distributed training in ML frameworks, and you're implementing it from scratch using Unix processes and pipes.

## The Solution: Ring Allreduce with Chunking

Arrange the N workers in a ring — each worker can only talk to its immediate right neighbor (send) and left neighbor (receive). The vector is split into N equal chunks. The algorithm runs in two phases:

### Phase 1 — Scatter-Reduce (N−1 rounds)

Each round, every worker sends one chunk to the right and receives one chunk from the left. When a chunk is received, it gets added element-wise into the worker's local copy. The chunk indices rotate each round so that after N−1 rounds, each worker holds the fully summed version of exactly one chunk (worker i owns chunk i).

### Phase 2 — Allgather (N−1 rounds)

Same ring, same direction, but now workers send their completed chunk around. On receive, the data is overwritten (no addition). After N−1 rounds, every worker has the complete reduced vector.

The result: all workers hold the same fully-summed vector, and at no point did any single process need to gather all raw data.

## Where Pipes Come In

This is a systems programming project — the interesting part is the plumbing, not the math.

**Ring pipes:** Each worker gets a write-fd to its right neighbor and a read-fd from its left neighbor. For N workers, that's N pipes forming a one-directional ring. The parent process creates all the pipes before forking, then each child closes the fds it doesn't own.

**Control pipes:** The parent also has a separate pipe pair to each worker — one for sending initial data to the worker (parent-to-worker), one for receiving the final result back (worker-to-parent). That's another 2N pipe endpoints.

**Total per worker:** 4 active file descriptors — ring-read, ring-write, parent-read, parent-write.

### Why this is hard

- After `fork()`, every child has copies of every fd. Each child must close all fds it doesn't use, or `read()` will never see EOF (because a write-end is still open somewhere), causing deadlocks.
- Pipes don't guarantee full writes or reads. A `write(fd, buf, 4096)` might only write 2000 bytes. You need `read_all` / `write_all` helper functions that loop until the full byte count is transferred.
- All N workers are running simultaneously. If one dies mid-phase, the others will hang on `read()` or get SIGPIPE on `write()`. You need to detect and handle this.

## Message Protocol

There are four message types flowing through the pipes:

| Message  | Direction       | Contents                                          |
|----------|-----------------|---------------------------------------------------|
| INIT     | Parent → Worker | Worker ID, N, vector length, float payload        |
| CHUNK    | Worker → Worker | Phase tag, chunk index, chunk length, float payload |
| DONE     | Worker → Parent | Worker ID, final reduced vector                   |
| SHUTDOWN | Parent → Worker | Tells workers to clean up on error                |

Each message is a fixed-size header struct followed by a variable-length float array. Define the structs in a shared header file so everyone agrees on the wire format.

## File Structure

| File             | Owner    | Description                                              |
|------------------|----------|----------------------------------------------------------|
| `main.c`         | Person A | Parent: arg parsing, pipe setup, fork, distribute, collect, verify |
| `worker.c/.h`    | Person B | Worker: receive INIT, run phases, send DONE              |
| `protocol.c/.h`  | Person C | Structs, read_all, write_all, send/recv helpers          |
| `Makefile`       | Person C | Build with -Wall -Wextra, no warnings                    |

## How to Split Between 3 People

**Person A — Parent / main.c:** Arg parsing, creating all pipes, forking N children, closing unused fds, sending INIT messages, collecting DONE messages with `select()` or a read loop, running `waitpid()`, optional correctness verification. This is the heaviest systems work.

**Person B — Worker / worker.c:** The worker entry point and main loop. Receive INIT, run scatter-reduce (N−1 rounds with the modular chunk-index math), run allgather (N−1 rounds), send DONE. The core algorithmic logic lives here.

**Person C — Protocol / protocol.c + Makefile:** Define all message structs, write `read_all()`, `write_all()`, `send_chunk()`, `recv_chunk()`, error-reporting utilities, SIGPIPE handling setup, and the Makefile. This person should finish the header file first — both A and B depend on these interfaces.

**Integration order:** C publishes `protocol.h` → A and B work in parallel → merge and test together.

## Sanity Check (N=3, L=6)

Use this to verify correctness by hand and in your code:

**Input:**
```
W0: [1, 1, 1, 1, 1, 1]
W1: [2, 2, 2, 2, 2, 2]
W2: [3, 3, 3, 3, 3, 3]
```

**Expected output at every worker:**
```
[6, 6, 6, 6, 6, 6]
```

Chunks (2 floats each): chunk 0 = [0,1], chunk 1 = [2,3], chunk 2 = [4,5].

After Phase 1, each worker owns one fully-reduced chunk:
```
W0's chunk 0 = [6, 6]
W1's chunk 1 = [6, 6]
W2's chunk 2 = [6, 6]
```

After Phase 2, all chunks are broadcast to everyone → full vector `[6, 6, 6, 6, 6, 6]` at all workers.

Have the parent independently compute the sum and compare against each worker's DONE payload. If they match, you're good.




## TINN 
Want to use the Tinn library in order to simulate distributed training in our code. 

We have ring allreduce implemented, but we want to make it so that we use it to simulate this distributed training using the library. The idea is that I don't think we change anything in the Tinn library itself, but if you look at test.c in the folder, we basically need to integrate this into our main.c. 

I think right now we mainly have three different workers. We can make that four, so we have four different GPUs.

What we do is we need to first download the data. I'm not sure how many rows there are. But we basically built in the test data as you can see. Then, after that, for every training iteration, what I'm thinking is that we should split it up into four different chunks and then we send the four different chunks to each child. Then, after each training iteration, we run the ring all-reduce algorithm. We first compute the gradient vector on each child and then we pass the gradient vectors around so that we add them all up and then we divide by four to find the average. This will allow us to simulate distributed training. We treat each different worker as a different GPU. 

I guess one thing is that, given the current architecture, we need to first send the data to the children, and I think we send it all at once. Actually, technically each worker has the entire matrix. Should we do the same in here, like we send each GPU the entire dataset, which we wouldn't normally do, but it only trains on a small, one-fourth chunk of that? 

Let's discuss this. Grill me with questions and anything that you want me to clarify.

See tinn/ for the tinn library and the main repo for everything else. My goal is to leave the tinn library untouched, we can copy tinn.c and tinn.h into main repo and then edit main.c. 