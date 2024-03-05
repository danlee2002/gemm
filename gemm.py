import time
import numpy as np
N = 2048

if __name__ == "__main__":
    # N^2 
    A = np.random.randn(N,N).astype(np.float32)
    # N^2
    B = np.random.randn(N,N).astype(np.float32)
    flop = 2* N * N * N
    print(f'{flop/1e9:.2f} GFLOP')
    for i in range(100):
        st = time.monotonic()
        C = A @ B 
        et = time.monotonic()
        duration = et - st 
        print(f'GFLOPS: {flop/duration * 1e-9}/s')


