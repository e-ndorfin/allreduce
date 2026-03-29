# Messages 
| Message  | Direction       | Contents                                          |
|----------|-----------------|---------------------------------------------------|
| INIT     | Parent → Worker | Worker ID, N, vector length, float payload        |
| CHUNK    | Worker → Worker | Phase tag, chunk index, chunk length, float payload |
| DONE     | Worker → Parent | Worker ID, final reduced vector                   |
| SHUTDOWN | Parent → Worker | Tells workers to clean up on error                |

How it would work is we would send CHUNK and then send an array of `vec_len`` floats after it. When we malloc the message and then try to say that the gradient vector length should be, say, 10, this doesn't work because we have to specify the array length pre-compilation - we can't do it at runtime. This is why we can't put the array of floats within the message. 

# Procedure 
**Initialization**
1. Parent makes children with `fork`
2. Use INIT message to send initial gradient vectors to each respective child

**Scatter-Reduce (N-1 rounds)**
1. Every worker sends one chunk to the right worker using CHUNK using `write` and receives a chunk from the left worker using `read`. The size of the chunk is (length of the gradient vector / number of workers). 
2. The flow should be in round 0, worker `i` sends chunk `i`, receives chunk `(i-1) mod N`, accumulates. 
2. When receiving a chunk, add to current running total
3. Repeat N-1 rounds, where N = number of workers / children

**Allgather**
1. Each worker sends the completed chunk that it has to the next worker and receives the completed chunk it doesn't have from the previous worker
2. After N-1 rounds, now each worker has a fully-summed vector!
3. Each worker can send result back to parent to verify correctness using DONE. 
4. Parent uses SHUTDOWN to close workers. 

