# Compilers
CC = gcc

# Flags
CFLAGS = -std=c11 -Wall -Wextra -g -O2 -ffast-math -ftree-vectorize -march=native -flto
TINY_LDFLAGS = -lm 
CG_LDFLAGS = -lm -lglfw -lGL -lGLEW

TARGETS = headless head

# Files
C_SOURCES = wtime.c photon.c xoshiro.c
C_OBJS = $(patsubst %.c, %.o, $(C_SOURCES))

headless: tiny_mc.o $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(TINY_LDFLAGS)

head: cg_mc.o $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(CG_LDFLAGS)

clean:
	rm -f $(TARGETS) *.o

cleangcda:
	rm -f *.gcda