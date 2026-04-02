CC = gcc
CFLAGS = -Wall -Wextra -g

OBJS = main.o protocol.o worker.o

all: allreduce

allreduce: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

%.o: %.c protocol.h worker.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o allreduce
