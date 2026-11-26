# Fast_triton_kmeans
Short and sweet, just a k means algo that runs fast on the gpu and is accurate.

 GENUINE ENGINE V11 MULTI-TASK DRIFT & REPROJECTION BENCHMARK (N=320,000 | D=1024 | K=64)
 Hardware: NVIDIA GeForce RTX 3060 | Base Fitter: Engine v11 (nosync)
```
>>> Task 1: Isolated Blobs + 20 Outliers
    v11 Initial Fit (t=0): 59.7 ms | Base MSE: 1105.7750
    Frozen v11 Codebook on X_drift: MSE = 1168.3431 (No update)
Method                               | Type | Fit ms  | MSE       | Raw Cos  | Raw Ovr% | Hung Cos (ms)    | Greedy (ms)
Triton Lloyd Reproj (1 pass, 16k)    | Warm | 2.73    | 1109.5182 | +0.998   | 100.0%   | +0.998 (0.62ms)  | +0.998 (2.12ms)
Triton Lloyd Reproj (2 passes, 16k)  | Warm | 3.52    | 1109.3992 | +0.998   | 100.0%   | +0.998 (0.46ms)  | +0.998 (2.13ms)
Triton Engine Reproj (1 step, 16k)   | Warm | 9.55    | 1108.8564 | +0.998   | 100.0%   | +0.998 (0.60ms)  | +0.998 (2.03ms)
Triton Engine Reproj (2 pass, 16k)   | Warm | 16.77   | 1108.9022 | +0.998   | 100.0%   | +0.998 (0.85ms)  | +0.998 (2.50ms)
CuBLAS Lloyd Warm (1 pass, full)     | Warm | 18.87   | 1104.7663 | +0.998   | 100.0%   | +0.998 (0.64ms)  | +0.998 (1.90ms)
Engine v11 Cold (from scratch)       | Cold | 58.73   | 1108.9307 | SCRAMBLED | 0.0%     | +0.998 (0.41ms)  | +0.998 (2.55ms)
e-means Cold (1.0 ep)                | Cold | 106.98  | 2128.6026 | SCRAMBLED | 0.0%     | +0.920 (0.49ms)  | +0.919 (3.07ms)
CuBLAS Lloyd Cold (15 iters)         | Cold | 260.49  | 5173.2740 | SCRAMBLED | 0.0%     | +0.692 (0.63ms)  | +0.690 (1.78ms)

>>> Task 2: L2-Normalized Manifold (Zipfian Hubs)
    v11 Initial Fit (t=0): 58.5 ms | Base MSE: 0.9072
    Frozen v11 Codebook on X_drift: MSE = 0.9073 (No update)
Method                               | Type | Fit ms  | MSE       | Raw Cos  | Raw Ovr% | Hung Cos (ms)    | Greedy (ms)
Triton Lloyd Reproj (1 pass, 16k)    | Warm | 2.07    | 0.9039    | +0.972   | 73.9%    | +0.972 (0.92ms)  | +0.972 (3.90ms)
Triton Lloyd Reproj (2 passes, 16k)  | Warm | 4.55    | 0.9025    | +0.938   | 62.6%    | +0.938 (0.86ms)  | +0.938 (2.52ms)
Triton Engine Reproj (1 step, 16k)   | Warm | 11.62   | 0.9030    | +0.930   | 59.8%    | +0.930 (1.10ms)  | +0.930 (1.95ms)
Triton Engine Reproj (2 pass, 16k)   | Warm | 18.99   | 0.9016    | +0.866   | 46.9%    | +0.866 (0.93ms)  | +0.866 (2.35ms)
CuBLAS Lloyd Warm (1 pass, full)     | Warm | 17.25   | 0.9035    | +0.989   | 85.2%    | +0.989 (0.68ms)  | +0.989 (1.89ms)
Engine v11 Cold (from scratch)       | Cold | 59.48   | 0.9069    | SCRAMBLED | 0.0%     | +0.418 (0.52ms)  | +0.393 (3.17ms)
e-means Cold (1.0 ep)                | Cold | 70.32   | 0.8963    | SCRAMBLED | 0.0%     | +0.428 (0.79ms)  | +0.418 (2.95ms)
CuBLAS Lloyd Cold (15 iters)         | Cold | 261.56  | 0.8970    | SCRAMBLED | 0.0%     | +0.419 (0.58ms)  | +0.405 (2.91ms)

>>> Task 3: Rotated Anisotropic Subspace
    v11 Initial Fit (t=0): 61.2 ms | Base MSE: 224.7096
    Frozen v11 Codebook on X_drift: MSE = 237.7614 (No update)
Method                               | Type | Fit ms  | MSE       | Raw Cos  | Raw Ovr% | Hung Cos (ms)    | Greedy (ms)
Triton Lloyd Reproj (1 pass, 16k)    | Warm | 2.45    | 224.6628  | +0.998   | 100.0%   | +0.998 (0.77ms)  | +0.998 (2.68ms)
Triton Lloyd Reproj (2 passes, 16k)  | Warm | 5.18    | 224.6795  | +0.998   | 100.0%   | +0.998 (0.86ms)  | +0.998 (2.60ms)
Triton Engine Reproj (1 step, 16k)   | Warm | 10.05   | 224.6720  | +0.998   | 100.0%   | +0.998 (0.62ms)  | +0.998 (1.70ms)
Triton Engine Reproj (2 pass, 16k)   | Warm | 16.08   | 224.6785  | +0.998   | 100.0%   | +0.998 (0.55ms)  | +0.998 (1.66ms)
CuBLAS Lloyd Warm (1 pass, full)     | Warm | 18.33   | 223.8092  | +0.998   | 100.0%   | +0.998 (0.87ms)  | +0.998 (2.68ms)
Engine v11 Cold (from scratch)       | Cold | 60.86   | 224.7181  | SCRAMBLED | 0.0%     | +0.998 (0.92ms)  | +0.998 (2.90ms)
e-means Cold (1.0 ep)                | Cold | 77.32   | 733.0153  | SCRAMBLED | 0.0%     | +0.823 (0.64ms)  | +0.819 (1.70ms)
CuBLAS Lloyd Cold (15 iters)         | Cold | 261.61  | 927.0401  | SCRAMBLED | 0.0%     | +0.753 (0.72ms)  | +0.746 (3.74ms)

>>> Task 4: 96% Core + 4% Heavy-Tail Clutter
    v11 Initial Fit (t=0): 64.6 ms | Base MSE: 5267.0983
    Frozen v11 Codebook on X_drift: MSE = 5302.5036 (No update)
Method                               | Type | Fit ms  | MSE       | Raw Cos  | Raw Ovr% | Hung Cos (ms)    | Greedy (ms)
Triton Lloyd Reproj (1 pass, 16k)    | Warm | 2.02    | 5277.0503 | +0.997   | 100.0%   | +0.997 (0.68ms)  | +0.997 (2.73ms)
Triton Lloyd Reproj (2 passes, 16k)  | Warm | 5.58    | 5276.1523 | +0.997   | 100.0%   | +0.997 (1.08ms)  | +0.997 (2.76ms)
Triton Engine Reproj (1 step, 16k)   | Warm | 8.20    | 5272.6937 | +0.998   | 100.0%   | +0.998 (0.99ms)  | +0.998 (2.74ms)
Triton Engine Reproj (2 pass, 16k)   | Warm | 18.16   | 5272.6321 | +0.998   | 100.0%   | +0.998 (0.54ms)  | +0.998 (2.24ms)
CuBLAS Lloyd Warm (1 pass, full)     | Warm | 16.75   | 5258.1066 | +0.998   | 100.0%   | +0.998 (0.49ms)  | +0.998 (2.23ms)
Engine v11 Cold (from scratch)       | Cold | 59.25   | 5271.2296 | SCRAMBLED | 0.0%     | +0.998 (0.46ms)  | +0.998 (1.61ms)
e-means Cold (1.0 ep)                | Cold | 94.00   | 5608.0277 | SCRAMBLED | 0.0%     | +0.921 (0.45ms)  | +0.920 (1.81ms)
CuBLAS Lloyd Cold (15 iters)         | Cold | 261.56  | 6629.7990 | SCRAMBLED | 0.0%     | +0.774 (0.90ms)  | +0.773 (2.05ms)

>>> Task 5: Extreme Imbalance (2 Giants + 62 Dwarfs)
    v11 Initial Fit (t=0): 59.4 ms | Base MSE: 7959.4681
    Frozen v11 Codebook on X_drift: MSE = 8018.2644 (No update)
Method                               | Type | Fit ms  | MSE       | Raw Cos  | Raw Ovr% | Hung Cos (ms)    | Greedy (ms)
Triton Lloyd Reproj (1 pass, 16k)    | Warm | 3.21    | 7959.4043 | +0.998   | 100.0%   | +0.998 (0.65ms)  | +0.998 (2.51ms)
Triton Lloyd Reproj (2 passes, 16k)  | Warm | 3.63    | 7959.7985 | +0.998   | 100.0%   | +0.998 (0.98ms)  | +0.998 (3.08ms)
Triton Engine Reproj (1 step, 16k)   | Warm | 8.71    | 7959.7610 | +0.998   | 100.0%   | +0.998 (0.62ms)  | +0.998 (1.69ms)
Triton Engine Reproj (2 pass, 16k)   | Warm | 18.77   | 7959.5239 | +0.998   | 100.0%   | +0.998 (0.73ms)  | +0.998 (1.88ms)
CuBLAS Lloyd Warm (1 pass, full)     | Warm | 16.92   | 7957.2683 | +0.998   | 100.0%   | +0.998 (0.45ms)  | +0.998 (2.57ms)
Engine v11 Cold (from scratch)       | Cold | 58.21   | 7959.6779 | SCRAMBLED | 0.0%     | +0.998 (0.68ms)  | +0.998 (2.03ms)
e-means Cold (1.0 ep)                | Cold | 91.40   | 9297.1538 | SCRAMBLED | 0.0%     | +0.611 (0.78ms)  | +0.609 (1.95ms)
CuBLAS Lloyd Cold (15 iters)         | Cold | 266.03  | 11060.9990 | SCRAMBLED | 0.0%     | +0.200 (1.00ms)  | +0.200 (3.16ms)
```

