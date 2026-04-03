CC = gcc
CFLAGS = -Wall -Wextra -g
LDFLAGS = -lm

OBJS = main.o protocol.o worker.o Tinn.o data.o

all: allreduce

allreduce: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c protocol.h worker.h Tinn.h data.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o allreduce
