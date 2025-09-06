import argparse
import csv
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd

def time_matmul(device, size, repeats):
    # Warm-up
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    torch.matmul(a, b)
    # Synchronize after warm-up
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

    # Timed runs
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        torch.matmul(a, b)
        # Accurate timing synchronization
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    return sum(times) / repeats


def main(args):
    # Determine devices
    devices = [('cpu', torch.device('cpu'))]
    if torch.cuda.is_available():
        devices.append(('cuda', torch.device('cuda')))
    if torch.backends.mps.is_available():
        devices.append(('mps', torch.device('mps')))

    # Prepare CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['device', 'matrix_size', 'time_s'])

        # Benchmark loop
        for power in range(2, args.max_power + 1):
            size = 2**power
            for name, dev in devices:
                avg_time = time_matmul(dev, size, args.repeats)
                print(f"{name:>4} | size={size:<6} avg_time={avg_time:.6f}s")
                writer.writerow([name, size, avg_time])

    # Plot results
    df = pd.read_csv(args.output)
    plt.figure(figsize=(8,5))
    for name in df['device'].unique():
        sub = df[df['device']==name]
        plt.plot(sub['matrix_size'], sub['time_s'], marker='o', label=name)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Matrix size (N x N)')
    plt.ylabel('Average time (s)')
    plt.title('MatMul Performance: CPU vs GPU vs Metal')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.savefig(args.plot, dpi=300)
    print(f"Results saved to {args.output} and plot to {args.plot}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark CPU, CUDA, and MPS matmul performance")
    parser.add_argument('--max_power', type=int, default=10,
                        help='Maximum exponent for matrix size (2**power)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='Number of repeats per size for averaging')
    parser.add_argument('--output', type=str, default='results.csv',
                        help='CSV filename for logging results')
    parser.add_argument('--plot', type=str, default='performance.png',
                        help='Filename for the output plot')
    args = parser.parse_args()
    main(args)
