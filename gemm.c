#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#define N 1024
#define BLOCK 4
float A[N][N];
// assuming B[n][n] is transposed
float B[N][N];
float C[N][N];

uint64_t nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

int main() {
  assert(N % BLOCK == 0);
  uint64_t start = nanos();

  for (int by = 0; by < N; by += BLOCK) {
    for (int bx = 0; bx < N; bx += BLOCK) {
      float tc[BLOCK][BLOCK];
      for (int y = 0; y < BLOCK; y++) {
        for (int x = 0; x < BLOCK; x++) {
          float acc = 0;
          // transposed dot product
          for (int k = 0; k < N; k++) {
            acc += A[by + y][k] * B[bx + x][k];
          }
          tc[y][x] = acc;
        }
      }

      for (int y = 0; y < BLOCK; y++) {
        for (int x = 0; x < BLOCK; x++) {
          C[bx + y][bx + x] = tc[y][x];
        }
      }
    }
  }

  uint64_t end = nanos();
  printf("%lld %lld\n", start, end);
  double s = (end - start) * 1e-9;
  double gflop = (2.0 * N * N * N) * 1e-9;
  printf("%0.2f GFLOP/s\n", gflop / s);
  return 0;
}
