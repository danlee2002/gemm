#include <assert.h>
#include <immintrin.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#define N 2048
#define BLOCK 4
#define THREADS 4
float A[N][N];
float B[N][N];
float C[N][N];
float transposedBlock[N][N];

void *dot(void *arg) {

  int value = (int)(intptr_t)arg;
  int start = N / THREADS;
  for (int by = N / THREADS * value; by < N / THREADS * (value + 1);
       by += BLOCK) {
    for (int bx = N / THREADS * value; bx < N / THREADS * (value + 1);
         bx += BLOCK) {
      float tc[BLOCK][BLOCK];
      for (int y = 0; y < BLOCK; y++) {
        for (int x = 0; x < BLOCK; x++) {
          __m256 acc = _mm256_setzero_ps();
          for (int k = 0; k < N; k++) {
            __m256 a = _mm256_load_ps(&A[by + y][k]);
            __m256 b = _mm256_load_ps(&transposedBlock[bx + y][k]);
            __m256 c = _mm256_mul_ps(a, b);
            acc = _mm256_add_ps(acc,c); 
          }
         _mm256_store_ps(&tc[y][x],acc);
        }
      }

      
    }
  }

  return NULL;
}

uint64_t nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

int main() {

  assert(N % BLOCK == 0);
  // naive implementation
  uint64_t start = nanos();
  for (int y = 0; y < N; y++) {
    for (int x = 0; x < N; x++) {
      float acc = 0;
      for (int k = 0; k < N; k++) {
        acc += A[y][k] * B[k][x];
      }
      C[x][y] = acc;
    }
  }
  uint64_t end = nanos();

  printf("naive approach:\n %lld %lld\n", start, end);
  double s = (end - start) * 1e-9;
  double gflop = (2.0 * N * N * N) * 1e-9;
  printf("%0.2f GFLOP/s\n", gflop / s);
  pthread_t threads[THREADS];

  start = nanos();
  // transpose matrix in c
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      transposedBlock[i][j] = B[j][i];
    }
  }
  for (int i = 0; i < THREADS; i++) {
    pthread_create(&threads[i], NULL, dot, (void *)(long)i);
  }
  for (int i = 0; i < THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
  end = nanos();
  printf("tiling approach: \n %lld %lld \n", start, end);
  s = (end - start) * 1e-9;
  gflop = (2.0 * N * N * N) * 1e-9;
  printf("%0.2f GFLOP/s\n", gflop / s);
  return 0;
}
