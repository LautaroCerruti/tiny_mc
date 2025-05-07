# Compilers
#CC = gcc
CC = icx
#CC = clang

# Flags
EXTRAFLAGS = 
#Extra flags gcc
#EXTRAFLAGS = -fopt-info-vec-optimized -fopt-info-vec-missed
#Extra flags icx
#EXTRAFLAGS = -qopt-report-phase=vec -qopt-report=5
#Extra flags clang
#EXTRAFLAGS = -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -fsave-optimization-record

# Flags gcc
#CFLAGS = -std=c11 -Wall -Wextra -Ofast -ffast-math -fdisable-tree-cunrolli -ftree-vectorize -march=native -flto 
# Flags icx
CFLAGS = -std=c11 -Wall -Wextra -O3 -ffast-math -vec -xHost -ipo -fno-unroll-loops
# Flags clang
#CFLAGS = -std=c11 -Wall -Wextra -Ofast -ffast-math -fvectorize -march=native -flto -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -fsave-optimization-record

TINY_LDFLAGS = -lm 
CG_LDFLAGS = -lm -lglfw -lGL -lGLEW

TARGETS = headless head

# Files
C_SOURCES = wtime.c photon.c xoshiro.c
C_OBJS = $(patsubst %.c, %.o, $(C_SOURCES))

headless: tiny_mc.o $(C_OBJS)
	$(CC) $(CFLAGS) $(EXTRAFLAGS) -o $@ $^ $(TINY_LDFLAGS)

head: cg_mc.o $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(CG_LDFLAGS)

clean:
	rm -f $(TARGETS) *.o

cleangcda:
	rm -f *.gcda