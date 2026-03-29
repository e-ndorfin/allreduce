CC = gcc
CFLAGS = -Wall -Wextra -g

OBJS = protocol.o

all: allreduce

allreduce: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

%.o: %.c protocol.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o allreduce
