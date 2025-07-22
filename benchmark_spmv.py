#!/usr/bin/env python3
"""
Comprehensive SpMV Implementation Benchmark Script
Downloads diverse matrices from Matrix Market and analyzes performance across implementations.
"""

import os
import sys
import subprocess
import urllib.request
import gzip
import time
import json
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("⚠️ matplotlib/numpy not available. Install with: pip install matplotlib numpy")

from pathlib import Path
import re

IMPLEMENTATIONS = [
    {"name": "gpu_mem", "description": "GPU Memory Optimized"},
    {"name": "cpu_csr", "description": "CPU Compressed Sparse Row"},
    {"name": "gpu_unrl", "description": "GPU Loop Unrolling"},
    {"name": "gpu_hbp", "description": "GPU Hash-Based Partitioning"}
]

# Benchmark output configuration
BENCHMARK_DIR = "benchmark"

class MatrixAnalyzer:
    def __init__(self):
        pass

    def analyze_matrix(self, filepath):
        """Extract matrix characteristics from MTX file"""
        try:
            with open(filepath, 'r') as f:
                # Skip comments
                line = f.readline()
                while line.startswith('%'):
                    line = f.readline()

                # Parse dimensions
                parts = line.strip().split()
                rows = int(parts[0])
                cols = int(parts[1])
                nnz = int(parts[2])

                # Calculate characteristics
                density = nnz / (rows * cols) if rows > 0 and cols > 0 else 0
                avg_nnz_per_row = nnz / rows if rows > 0 else 0

                return {
                    "rows": rows,
                    "cols": cols,
                    "nnz": nnz,
                    "density": density,
                    "avg_nnz_per_row": avg_nnz_per_row,
                    "sparsity_ratio": 1.0 - density
                }
        except Exception as e:
            print(f"❌ Failed to analyze {filepath}: {e}")
            return None

class SpMVBenchmark:
    def __init__(self):
        self.results = {}

    def build_implementation(self, impl_name):
        """Build specific implementation"""
        try:
            cmd = f"make spmv_{impl_name}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Built {impl_name}")
                return True
            else:
                print(f"❌ Failed to build {impl_name}: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error building {impl_name}: {e}")
            return False

    def run_benchmark(self, impl_name, matrix_path, timeout=120):
        """Run benchmark for specific implementation and matrix"""
        executable = f"./spmv_{impl_name}"

        if not os.path.exists(executable):
            print(f"❌ Executable {executable} not found")
            return None

        try:
            cmd = f"timeout {timeout} {executable} {matrix_path}"
            start_time = time.time()
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            end_time = time.time()

            if result.returncode == 0:
                # Parse output for timing information
                output = result.stdout
                elapsed_match = re.search(r'Elapsed time: ([0-9.]+) seconds', output)
                bandwidth_match = re.search(r'Effective bandwidth: ([0-9.]+) GB/s', output)

                if elapsed_match:
                    elapsed_time = float(elapsed_match.group(1))
                    parsed_bandwidth = float(bandwidth_match.group(1)) if bandwidth_match else 0.0

                    # Calculate theoretical bandwidth for validation
                    # Get matrix info for enhanced bandwidth calculation
                    analyzer = MatrixAnalyzer()
                    matrix_info = analyzer.analyze_matrix(matrix_path)

                    theoretical_bandwidth = 0.0
                    if matrix_info:
                        # Memory accesses: CSR arrays + input vector + output vector
                        # For SpMV: 2*nnz reads (val, col) + row reads + 2*vector accesses
                        total_bytes = (
                            matrix_info['nnz'] * (4 + 4) +  # val (float) + col (int) arrays
                            (matrix_info['rows'] + 1) * 4 +  # row array (int)
                            matrix_info['cols'] * 4 +        # input vector (float)
                            matrix_info['rows'] * 4          # output vector (float)
                        )
                        theoretical_bandwidth = total_bytes / (elapsed_time * 1e9)  # GB/s

                    # Use parsed bandwidth if available and reasonable, otherwise use theoretical
                    final_bandwidth = parsed_bandwidth if parsed_bandwidth > 0 else theoretical_bandwidth

                    # Validate bandwidth (sanity check)
                    if final_bandwidth > 1000:  # > 1TB/s seems unrealistic
                        print(f"⚠️ Suspiciously high bandwidth {final_bandwidth:.2f} GB/s for {impl_name}")
                        final_bandwidth = min(final_bandwidth, theoretical_bandwidth)

                    return {
                        "success": True,
                        "elapsed_time": elapsed_time,
                        "bandwidth": final_bandwidth,
                        "theoretical_bandwidth": theoretical_bandwidth,
                        "parsed_bandwidth": parsed_bandwidth,
                        "wall_time": end_time - start_time
                    }
                else:
                    print(f"⚠️ Could not parse timing from {impl_name} output")
                    return {
                        "success": True,
                        "elapsed_time": end_time - start_time,
                        "bandwidth": 0.0,
                        "wall_time": end_time - start_time
                    }
            else:
                print(f"❌ {impl_name} failed: {result.stderr}")
                return {"success": False, "error": result.stderr}

        except Exception as e:
            print(f"❌ Error running {impl_name}: {e}")
            return {"success": False, "error": str(e)}

    def benchmark_all(self, matrices):
        """Run benchmarks for all implementations and matrices"""
        print("\n🔨 Building implementations...")
        built_impls = []
        for impl in IMPLEMENTATIONS:
            if self.build_implementation(impl["name"]):
                built_impls.append(impl)

        print(f"\n🚀 Starting benchmarks with {len(built_impls)} implementations...")

        for matrix_idx, (config, filepath) in enumerate(matrices):
            matrix_name = config["name"]
            progress = f"({matrix_idx+1}/{len(matrices)})"
            print(f"\n📊 {progress} Benchmarking matrix: {matrix_name}")

            # Analyze matrix first
            analyzer = MatrixAnalyzer()
            matrix_info = analyzer.analyze_matrix(filepath)

            if matrix_info:
                print(f"   Size: {matrix_info['rows']}×{matrix_info['cols']}")
                print(f"   Non-zeros: {matrix_info['nnz']:,}")
                print(f"   Density: {matrix_info['density']:.6f}")
                print(f"   Avg NNZ/row: {matrix_info['avg_nnz_per_row']:.2f}")

            self.results[matrix_name] = {
                "config": config,
                "matrix_info": matrix_info,
                "implementations": {}
            }

            # Benchmark each implementation
            for idx, impl in enumerate(built_impls):
                impl_name = impl["name"]
                progress = f"[{idx+1}/{len(built_impls)}]"
                print(f"   {progress} Running {impl_name}...", end=" ", flush=True)

                result = self.run_benchmark(impl_name, filepath)
                self.results[matrix_name]["implementations"][impl_name] = {
                    "result": result,
                    "description": impl["description"]
                }

                if result and result["success"]:
                    print(f"✅ {result['elapsed_time']:.6f}s, {result['bandwidth']:.2f} GB/s")
                else:
                    print("❌ Failed")

    def save_results(self, filename="benchmark_results.json"):
        """Save results to JSON file"""
        # Ensure benchmark directory exists
        benchmark_dir = Path(BENCHMARK_DIR)
        benchmark_dir.mkdir(exist_ok=True)

        # Save to benchmark directory
        filepath = benchmark_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Results saved to {filepath}")

        # Also save a human-readable summary report
        self.save_summary_report(benchmark_dir)

    def save_summary_report(self, benchmark_dir):
        """Save a detailed text summary report"""
        report_path = benchmark_dir / "benchmark_summary_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SpMV BENCHMARK RESULTS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Write configuration
            f.write("BENCHMARK CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write("Implementations tested:\n")
            for impl in IMPLEMENTATIONS:
                f.write(f"  - {impl['name']}: {impl['description']}\n")
            f.write(f"\nTotal matrices tested: {len(self.results)}\n\n")

            # Matrix categorization
            f.write("MATRIX CATEGORIZATION:\n")
            f.write("-" * 40 + "\n")

            categories = {'grid': [], 'random': [], 'band': [], 'power': []}
            for matrix_name in self.results:
                if 'grid' in matrix_name:
                    categories['grid'].append(matrix_name)
                elif 'random' in matrix_name:
                    categories['random'].append(matrix_name)
                elif 'band' in matrix_name:
                    categories['band'].append(matrix_name)
                elif 'power' in matrix_name:
                    categories['power'].append(matrix_name)

            for cat, matrices in categories.items():
                if matrices:
                    f.write(f"  {cat.upper()} matrices ({len(matrices)}): {', '.join(matrices)}\n")
            f.write("\n")

            # Overall performance summary
            f.write("OVERALL PERFORMANCE SUMMARY:\n")
            f.write("-" * 40 + "\n")

            impl_stats = {}
            for matrix_name in self.results:
                for impl_name in self.results[matrix_name]["implementations"]:
                    result = self.results[matrix_name]["implementations"][impl_name]["result"]
                    if result and result["success"]:
                        if impl_name not in impl_stats:
                            impl_stats[impl_name] = {'times': [], 'bandwidths': [], 'wins': 0}
                        impl_stats[impl_name]['times'].append(result["elapsed_time"])
                        impl_stats[impl_name]['bandwidths'].append(result["bandwidth"])

            # Count wins (fastest per matrix)
            for matrix_name in self.results:
                best_time = float('inf')
                best_impl = None
                for impl_name in self.results[matrix_name]["implementations"]:
                    result = self.results[matrix_name]["implementations"][impl_name]["result"]
                    if result and result["success"] and result["elapsed_time"] < best_time:
                        best_time = result["elapsed_time"]
                        best_impl = impl_name
                if best_impl and best_impl in impl_stats:
                    impl_stats[best_impl]['wins'] += 1

            for impl_name, stats in impl_stats.items():
                if stats['times']:
                    avg_time = sum(stats['times']) / len(stats['times'])
                    avg_bw = sum(stats['bandwidths']) / len(stats['bandwidths'])
                    f.write(f"  {impl_name:15} - Avg Time: {avg_time:.6f}s  "
                           f"Avg BW: {avg_bw:6.2f} GB/s  Wins: {stats['wins']}/{len(self.results)}\n")
            f.write("\n")

            # Write detailed matrix information and results
            for matrix_name in sorted(self.results.keys()):
                config = self.results[matrix_name]["config"]
                info = self.results[matrix_name]["matrix_info"]

                f.write(f"MATRIX: {matrix_name.upper()}\n")
                f.write("-" * 40 + "\n")
                # f.write(f"Description: {config['description']}\n")

                if info:
                    f.write(f"Size: {info['rows']:,} × {info['cols']:,} = {info['rows']*info['cols']:,} elements\n")
                    f.write(f"Non-zeros: {info['nnz']:,} ({info['density']:.2e} density)\n")
                    f.write(f"Sparsity: {(1-info['density'])*100:.4f}%\n")
                    f.write(f"Average NNZ per row: {info['avg_nnz_per_row']:.2f}\n")

                f.write("\nImplementation Results (sorted by execution time):\n")

                results_list = []
                for impl_name in self.results[matrix_name]["implementations"]:
                    impl_data = self.results[matrix_name]["implementations"][impl_name]
                    result = impl_data["result"]
                    desc = impl_data["description"]

                    if result and result["success"]:
                        results_list.append({
                            'name': impl_name,
                            'desc': desc,
                            'time': result["elapsed_time"],
                            'bandwidth': result["bandwidth"],
                            'theo_bw': result.get("theoretical_bandwidth", 0.0)
                        })

                # Sort by execution time (fastest first)
                results_list.sort(key=lambda x: x['time'])

                for i, res in enumerate(results_list):
                    rank = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1:2}."
                    speedup = results_list[0]['time'] / res['time'] if res['time'] > 0 else 1.0
                    efficiency = (res['bandwidth'] / res['theo_bw'] * 100) if res['theo_bw'] > 0 else 0
                    f.write(f"  {rank} {res['name']:12} - {res['time']:8.6f}s  "
                           f"BW: {res['bandwidth']:6.2f} GB/s  "
                           f"Speedup: {speedup:5.2f}x  Eff: {efficiency:5.1f}%\n")

                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("NOTES:\n")
            f.write("- Bandwidth calculation includes all memory accesses (matrix + vectors)\n")
            f.write("- Efficiency = (Actual BW / Theoretical BW) * 100%\n")
            f.write("- Speedup calculated relative to fastest implementation per matrix\n")
            f.write("- Wins = number of matrices where implementation was fastest\n")
            f.write("=" * 80 + "\n")
            f.write("End of Report\n")

        print(f"📄 Enhanced summary report saved to {report_path}")

class ResultsVisualizer:
    def __init__(self, results):
        self.results = results

    def create_performance_plots(self):
        """Create comprehensive performance visualization"""
        if not HAS_PLOTTING:
            print("⚠️ Plotting libraries not available")
            return

        # Set up the plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                print("Using default matplotlib style")

        # Ensure benchmark directory exists
        benchmark_dir = Path(BENCHMARK_DIR)
        benchmark_dir.mkdir(exist_ok=True)

        # Prepare data
        matrices = list(self.results.keys())
        implementations = []

        # Get all successful implementations
        for matrix_name in matrices:
            for impl_name in self.results[matrix_name]["implementations"]:
                result = self.results[matrix_name]["implementations"][impl_name]["result"]
                if result and result["success"] and impl_name not in implementations:
                    implementations.append(impl_name)

        # Create comprehensive plot with improved layout
        fig = plt.figure(figsize=(24, 16))

        # Plot 1: Execution Time Comparison
        plt.subplot(3, 3, 1)
        self._plot_execution_times(matrices, implementations)

        # Plot 2: Bandwidth Comparison
        plt.subplot(3, 3, 2)
        self._plot_bandwidth(matrices, implementations)

        # Plot 3: Speedup Analysis
        plt.subplot(3, 3, 3)
        self._plot_speedup(matrices, implementations)

        # Plot 4: Performance vs Matrix Size (Connected)
        plt.subplot(3, 3, 4)
        self._plot_performance_vs_size(matrices, implementations)

        # Plot 5: Bandwidth vs Matrix Size (Connected)
        plt.subplot(3, 3, 5)
        self._plot_bandwidth_vs_size(matrices, implementations)

        # Plot 6: Computational Efficiency
        plt.subplot(3, 3, 6)
        self._plot_efficiency_analysis(matrices, implementations)

        # Plot 7: Matrix Characteristics
        plt.subplot(3, 3, 7)
        self._plot_matrix_characteristics(matrices)

        # Plot 8-9: Create a combined matrix analysis subplot
        plt.subplot(3, 3, 8)
        self._plot_matrix_size_distribution(matrices)

        plt.subplot(3, 3, 9)
        self._plot_implementation_summary(matrices, implementations)

        plt.tight_layout(pad=3.0)

        # Save comprehensive plot
        filepath = benchmark_dir / 'spmv_benchmark_analysis.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"📊 Comprehensive performance plots saved to {filepath}")
        plt.close()

        # Create and save individual plots
        self._save_individual_plots(matrices, implementations, benchmark_dir)
        plt.show() if HAS_PLOTTING else None

    def _plot_execution_times(self, matrices, implementations):
        """Plot execution time comparison"""
        x = np.arange(len(matrices))
        width = 0.8 / len(implementations)

        for i, impl in enumerate(implementations):
            times = []
            for matrix in matrices:
                result = self.results[matrix]["implementations"].get(impl, {}).get("result")
                if result and result["success"]:
                    times.append(result["elapsed_time"])
                else:
                    times.append(0)

            plt.bar(x + i * width, times, width, label=impl, alpha=0.8)

        plt.xlabel('Matrix')
        plt.ylabel('Execution Time (s)')
        plt.title('Execution Time Comparison')
        plt.xticks(x + width * (len(implementations) - 1) / 2, matrices, rotation=45)
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

    def _plot_bandwidth(self, matrices, implementations):
        """Plot bandwidth comparison"""
        x = np.arange(len(matrices))
        width = 0.8 / len(implementations)

        for i, impl in enumerate(implementations):
            bandwidths = []
            for matrix in matrices:
                result = self.results[matrix]["implementations"].get(impl, {}).get("result")
                if result and result["success"]:
                    bandwidths.append(result["bandwidth"])
                else:
                    bandwidths.append(0)

            plt.bar(x + i * width, bandwidths, width, label=impl, alpha=0.8)

        plt.xlabel('Matrix')
        plt.ylabel('Bandwidth (GB/s)')
        plt.title('Memory Bandwidth Utilization')
        plt.xticks(x + width * (len(implementations) - 1) / 2, matrices, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

    def _plot_speedup(self, matrices, implementations):
        """Plot speedup relative to baseline"""
        if not implementations:
            return

        baseline_impl = implementations[0]  # Use first implementation as baseline
        x = np.arange(len(matrices))
        width = 0.8 / len(implementations)

        for i, impl in enumerate(implementations):
            speedups = []
            for matrix in matrices:
                baseline_result = self.results[matrix]["implementations"].get(baseline_impl, {}).get("result")
                current_result = self.results[matrix]["implementations"].get(impl, {}).get("result")

                if (baseline_result and baseline_result["success"] and
                    current_result and current_result["success"]):
                    speedup = baseline_result["elapsed_time"] / current_result["elapsed_time"]
                    speedups.append(speedup)
                else:
                    speedups.append(0)

            plt.bar(x + i * width, speedups, width, label=f'{impl} vs {baseline_impl}', alpha=0.8)

        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline')
        plt.xlabel('Matrix')
        plt.ylabel('Speedup')
        plt.title(f'Speedup Analysis (vs {baseline_impl})')
        plt.xticks(x + width * (len(implementations) - 1) / 2, matrices, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

    def _plot_matrix_characteristics(self, matrices):
        """Plot matrix characteristics"""
        characteristics = {
            'rows': [],
            'nnz': [],
            'density': [],
            'avg_nnz_per_row': []
        }

        for matrix in matrices:
            info = self.results[matrix]["matrix_info"]
            if info:
                characteristics['rows'].append(info['rows'])
                characteristics['nnz'].append(info['nnz'])
                characteristics['density'].append(info['density'])
                characteristics['avg_nnz_per_row'].append(info['avg_nnz_per_row'])

        x = np.arange(len(matrices))

        # Create secondary y-axis for density
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        bars1 = ax1.bar(x - 0.2, characteristics['avg_nnz_per_row'], 0.4,
                       label='Avg NNZ/row', alpha=0.8, color='skyblue')
        bars2 = ax2.bar(x + 0.2, characteristics['density'], 0.4,
                       label='Density', alpha=0.8, color='lightcoral')

        ax1.set_xlabel('Matrix')
        ax1.set_ylabel('Average NNZ per Row', color='skyblue')
        ax2.set_ylabel('Density', color='lightcoral')
        ax1.set_title('Matrix Characteristics')
        ax1.set_xticks(x)
        ax1.set_xticklabels(matrices, rotation=45)

        # Add size annotations
        for i, matrix in enumerate(matrices):
            info = self.results[matrix]["matrix_info"]
            if info:
                ax1.text(i, characteristics['avg_nnz_per_row'][i],
                        f"{info['rows']}×{info['cols']}\n{info['nnz']:,} NNZ",
                        ha='center', va='bottom', fontsize=8)

        ax1.grid(True, alpha=0.3)

    def _plot_performance_vs_size(self, matrices, implementations):
        """Plot performance vs matrix size"""
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']

        for i, impl in enumerate(implementations):
            sizes = []
            times = []
            matrix_names = []

            for matrix in matrices:
                info = self.results[matrix]["matrix_info"]
                result = self.results[matrix]["implementations"].get(impl, {}).get("result")

                if info and result and result["success"]:
                    size = info['rows'] * info['cols']
                    sizes.append(size)
                    times.append(result["elapsed_time"])
                    matrix_names.append(matrix)

            if sizes and times:
                # Sort by matrix size for proper line connection
                sorted_data = sorted(zip(sizes, times, matrix_names), key=lambda x: x[0])
                sorted_sizes, sorted_times, sorted_names = zip(*sorted_data)

                color = colors[i % len(colors)]
                # Plot connected lines and points
                plt.plot(sorted_sizes, sorted_times, 'o-', color=color, label=impl,
                        alpha=0.8, linewidth=2, markersize=6, markerfacecolor=color,
                        markeredgecolor='white', markeredgewidth=1)

        plt.xlabel('Matrix Size (rows × cols)')
        plt.ylabel('Execution Time (s)')
        plt.title('Performance vs Matrix Size (Connected)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def _plot_bandwidth_vs_size(self, matrices, implementations):
        """Plot bandwidth vs matrix size with connected lines"""
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']

        for i, impl in enumerate(implementations):
            sizes = []
            bandwidths = []
            matrix_names = []

            for matrix in matrices:
                info = self.results[matrix]["matrix_info"]
                result = self.results[matrix]["implementations"].get(impl, {}).get("result")

                if info and result and result["success"]:
                    size = info['rows'] * info['cols']
                    sizes.append(size)
                    bandwidths.append(result["bandwidth"])
                    matrix_names.append(matrix)

            if sizes and bandwidths:
                # Sort by matrix size for proper line connection
                sorted_data = sorted(zip(sizes, bandwidths, matrix_names), key=lambda x: x[0])
                sorted_sizes, sorted_bandwidths, sorted_names = zip(*sorted_data)

                color = colors[i % len(colors)]
                # Plot connected lines and points
                plt.plot(sorted_sizes, sorted_bandwidths, 'o-', color=color, label=impl,
                        alpha=0.8, linewidth=2, markersize=6, markerfacecolor=color,
                        markeredgecolor='white', markeredgewidth=1)

        plt.xlabel('Matrix Size (rows × cols)')
        plt.ylabel('Bandwidth (GB/s)')
        plt.title('Memory Bandwidth vs Matrix Size (Connected)')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def _plot_matrix_size_distribution(self, matrices):
        """Plot matrix size distribution"""
        sizes = []
        names = []

        for matrix in matrices:
            info = self.results[matrix]["matrix_info"]
            if info:
                sizes.append(info['rows'] * info['cols'])
                names.append(matrix[:15] + '...' if len(matrix) > 15 else matrix)

        if sizes:
            plt.barh(range(len(names)), sizes, alpha=0.7, color='steelblue')
            plt.xlabel('Matrix Size (rows × cols)')
            plt.ylabel('Matrix')
            plt.title('Matrix Size Distribution')
            plt.yticks(range(len(names)), names, fontsize=8)
            plt.xscale('log')
            plt.grid(True, alpha=0.3, axis='x')

    def _plot_implementation_summary(self, matrices, implementations):
        """Plot implementation performance summary"""
        if not implementations:
            return

        # Calculate average performance across all matrices
        avg_times = []
        avg_bandwidths = []

        for impl in implementations:
            times = []
            bandwidths = []

            for matrix in matrices:
                result = self.results[matrix]["implementations"].get(impl, {}).get("result")
                if result and result["success"]:
                    times.append(result["elapsed_time"])
                    bandwidths.append(result["bandwidth"])

            avg_times.append(np.mean(times) if times else 0)
            avg_bandwidths.append(np.mean(bandwidths) if bandwidths else 0)

        # Create dual-axis plot
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        x = np.arange(len(implementations))

        bars1 = ax1.bar(x - 0.2, avg_times, 0.4, label='Avg Time (s)',
                       alpha=0.8, color='lightcoral')
        bars2 = ax2.bar(x + 0.2, avg_bandwidths, 0.4, label='Avg Bandwidth (GB/s)',
                       alpha=0.8, color='lightblue')

        ax1.set_xlabel('Implementation')
        ax1.set_ylabel('Average Time (s)', color='red')
        ax2.set_ylabel('Average Bandwidth (GB/s)', color='blue')
        ax1.set_title('Implementation Performance Summary')
        ax1.set_xticks(x)
        ax1.set_xticklabels([impl[:8] for impl in implementations], rotation=45)
        ax1.set_yscale('log')

        # Add value labels on bars
        for i, (t, b) in enumerate(zip(avg_times, avg_bandwidths)):
            if t > 0:
                ax1.text(i - 0.2, t, f'{t:.4f}', ha='center', va='bottom', fontsize=8)
            if b > 0:
                ax2.text(i + 0.2, b, f'{b:.1f}', ha='center', va='bottom', fontsize=8)

        ax1.grid(True, alpha=0.3)

    def _plot_detailed_matrix_analysis(self, matrices):
        """Plot detailed analysis of matrix properties"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Collect data
        matrix_data = []
        for matrix in matrices:
            info = self.results[matrix]["matrix_info"]
            if info:
                matrix_data.append({
                    'name': matrix,
                    'size': info['rows'] * info['cols'],
                    'nnz': info['nnz'],
                    'density': info['density'],
                    'avg_nnz_per_row': info['avg_nnz_per_row'],
                    'rows': info['rows'],
                    'cols': info['cols']
                })

        # Sort by matrix size
        matrix_data.sort(key=lambda x: x['size'])

        names = [d['name'] for d in matrix_data]
        sizes = [d['size'] for d in matrix_data]
        nnzs = [d['nnz'] for d in matrix_data]
        densities = [d['density'] for d in matrix_data]
        avg_nnz_per_rows = [d['avg_nnz_per_row'] for d in matrix_data]

        # Plot 1: Matrix size vs NNZ
        ax1.loglog(sizes, nnzs, 'bo-', markersize=6, linewidth=2)
        ax1.set_xlabel('Matrix Size (rows × cols)')
        ax1.set_ylabel('Number of Non-zeros')
        ax1.set_title('Matrix Size vs Non-zeros')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Density analysis
        ax2.semilogx(sizes, densities, 'ro-', markersize=6, linewidth=2)
        ax2.set_xlabel('Matrix Size (rows × cols)')
        ax2.set_ylabel('Density')
        ax2.set_title('Matrix Size vs Density')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Average NNZ per row
        ax3.semilogx(sizes, avg_nnz_per_rows, 'go-', markersize=6, linewidth=2)
        ax3.set_xlabel('Matrix Size (rows × cols)')
        ax3.set_ylabel('Average NNZ per Row')
        ax3.set_title('Matrix Size vs Avg NNZ per Row')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Matrix type distribution
        matrix_types = {}
        for d in matrix_data:
            matrix_type = d['name'].split('_')[1] if '_' in d['name'] else 'other'
            if matrix_type not in matrix_types:
                matrix_types[matrix_type] = []
            matrix_types[matrix_type].append(d['size'])

        type_names = list(matrix_types.keys())
        type_counts = [len(matrix_types[t]) for t in type_names]
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'orange']

        ax4.pie(type_counts, labels=type_names, autopct='%1.1f%%',
               colors=colors[:len(type_names)], startangle=90)
        ax4.set_title('Matrix Type Distribution')

        plt.tight_layout()

    def _plot_efficiency_analysis(self, matrices, implementations):
        """Plot efficiency analysis (GFLOPS)"""
        x = np.arange(len(matrices))
        width = 0.8 / len(implementations)

        for i, impl in enumerate(implementations):
            gflops = []
            for matrix in matrices:
                info = self.results[matrix]["matrix_info"]
                result = self.results[matrix]["implementations"].get(impl, {}).get("result")

                if info and result and result["success"]:
                    # Approximate GFLOPS: 2 * nnz / (time * 1e9)
                    flops = 2.0 * info['nnz'] / (result["elapsed_time"] * 1e9)
                    gflops.append(flops)
                else:
                    gflops.append(0)

            plt.bar(x + i * width, gflops, width, label=impl, alpha=0.8)

        plt.xlabel('Matrix')
        plt.ylabel('Performance (GFLOPS)')
        plt.title('Computational Efficiency')
        plt.xticks(x + width * (len(implementations) - 1) / 2, matrices, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

    def print_summary(self):
        """Print detailed performance summary"""
        print("\n" + "="*80)
        print("📊 SPMV BENCHMARK RESULTS SUMMARY")
        print("="*80)

        for matrix_name in self.results:
            config = self.results[matrix_name]["config"]
            info = self.results[matrix_name]["matrix_info"]

            print(f"\n🔍 Matrix: {matrix_name}")
            # print(f"   Description: {config['description']}")
            if info:
                print(f"   Size: {info['rows']:,} × {info['cols']:,}")
                print(f"   Non-zeros: {info['nnz']:,}")
                print(f"   Density: {info['density']:.2e}")
                print(f"   Avg NNZ/row: {info['avg_nnz_per_row']:.2f}")

            print(f"   Implementation Results:")

            results_list = []
            for impl_name in self.results[matrix_name]["implementations"]:
                impl_data = self.results[matrix_name]["implementations"][impl_name]
                result = impl_data["result"]
                desc = impl_data["description"]

                if result and result["success"]:
                    results_list.append({
                        'name': impl_name,
                        'desc': desc,
                        'time': result["elapsed_time"],
                        'bandwidth': result["bandwidth"]
                    })

            # Sort by execution time
            results_list.sort(key=lambda x: x['time'])

            for i, res in enumerate(results_list):
                status = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📈"
                speedup = results_list[0]['time'] / res['time'] if res['time'] > 0 else 1.0
                print(f"   {status} {res['name']:15} ({res['desc']:25}) "
                      f"Time: {res['time']:.6f}s  BW: {res['bandwidth']:6.2f} GB/s  "
                      f"Speedup: {speedup:.2f}x")

        print("\n" + "="*80)

    def _save_individual_plots(self, matrices, implementations, benchmark_dir):
        """Save individual plots as separate PNG files"""
        plot_configs = [
            ("execution_times", "Execution Time Comparison", self._plot_execution_times),
            ("bandwidth", "Memory Bandwidth Utilization", self._plot_bandwidth),
            ("bandwidth_vs_size", "Bandwidth vs Matrix Size", self._plot_bandwidth_vs_size),
            ("speedup", "Speedup Analysis", self._plot_speedup),
            ("matrix_characteristics", "Matrix Characteristics", self._plot_matrix_characteristics),
            ("performance_vs_size", "Performance vs Matrix Size", self._plot_performance_vs_size),
            ("efficiency", "Computational Efficiency", self._plot_efficiency_analysis),
            ("detailed_matrix_analysis", "Detailed Matrix Analysis", self._plot_detailed_matrix_analysis)
        ]

        for filename, title, plot_func in plot_configs:
            try:
                plt.figure(figsize=(12, 8))

                if filename in ["matrix_characteristics", "detailed_matrix_analysis"]:
                    plot_func(matrices)
                else:
                    plot_func(matrices, implementations)

                plt.title(title, fontsize=16, fontweight='bold')
                plt.tight_layout()

                filepath = benchmark_dir / f'spmv_{filename}.png'
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"📊 {title} saved to {filepath}")
                plt.close()

            except Exception as e:
                print(f"⚠️ Error saving {title}: {e}")
                plt.close()

    def create_ascii_charts(self):
        """Create ASCII-based performance charts"""
        print("\n" + "="*80)
        print("📊 ASCII PERFORMANCE CHARTS")
        print("="*80)

        matrices = list(self.results.keys())
        implementations = []

        # Get all successful implementations
        for matrix_name in matrices:
            for impl_name in self.results[matrix_name]["implementations"]:
                result = self.results[matrix_name]["implementations"][impl_name]["result"]
                if result and result["success"] and impl_name not in implementations:
                    implementations.append(impl_name)

        if not implementations:
            print("No successful implementations to chart.")
            return

        # Chart 1: Execution Time Comparison
        print("\n📈 EXECUTION TIME COMPARISON")
        print("-" * 60)

        for matrix in matrices:
            print(f"\n{matrix}:")
            times = []
            labels = []

            for impl in implementations:
                result = self.results[matrix]["implementations"].get(impl, {}).get("result")
                if result and result["success"]:
                    times.append(result["elapsed_time"])
                    labels.append(impl)

            if times:
                max_time = max(times)
                for i, (label, time) in enumerate(zip(labels, times)):
                    bar_length = int((time / max_time) * 40) if max_time > 0 else 0
                    bar = "█" * bar_length + "░" * (40 - bar_length)
                    print(f"  {label:15} |{bar}| {time:.6f}s")

        # Chart 2: Bandwidth Comparison
        print("\n📊 BANDWIDTH COMPARISON")
        print("-" * 60)

        for matrix in matrices:
            print(f"\n{matrix}:")
            bandwidths = []
            labels = []

            for impl in implementations:
                result = self.results[matrix]["implementations"].get(impl, {}).get("result")
                if result and result["success"]:
                    bandwidths.append(result["bandwidth"])
                    labels.append(impl)

            if bandwidths:
                max_bw = max(bandwidths) if max(bandwidths) > 0 else 1
                for i, (label, bw) in enumerate(zip(labels, bandwidths)):
                    bar_length = int((bw / max_bw) * 40) if max_bw > 0 else 0
                    bar = "█" * bar_length + "░" * (40 - bar_length)
                    print(f"  {label:15} |{bar}| {bw:6.2f} GB/s")

        # Chart 3: Speedup Analysis
        if len(implementations) > 1:
            baseline_impl = implementations[0]
            print(f"\n⚡ SPEEDUP ANALYSIS (vs {baseline_impl})")
            print("-" * 60)

            for matrix in matrices:
                print(f"\n{matrix}:")
                baseline_result = self.results[matrix]["implementations"].get(baseline_impl, {}).get("result")

                if baseline_result and baseline_result["success"]:
                    baseline_time = baseline_result["elapsed_time"]

                    for impl in implementations:
                        result = self.results[matrix]["implementations"].get(impl, {}).get("result")
                        if result and result["success"]:
                            speedup = baseline_time / result["elapsed_time"] if result["elapsed_time"] > 0 else 1.0

                            # Create speedup bar (scale: 1x = 20 chars, max 3x = 60 chars)
                            bar_length = min(int(speedup * 20), 60)
                            bar = "█" * bar_length
                            status = "🚀" if speedup > 1.2 else "📊" if speedup > 0.8 else "🐌"

                            print(f"  {impl:15} |{bar:<60}| {speedup:.2f}x {status}")

        print("\n" + "="*80)

def main():
    """Main benchmark execution"""
    print("🚀 SpMV Implementation Benchmark Suite")
    print("="*50)

    # Create benchmark directory
    benchmark_dir = Path(BENCHMARK_DIR)
    benchmark_dir.mkdir(exist_ok=True)
    print(f"📁 Benchmark directory created: {benchmark_dir.resolve()}")

    folder = "resources"
    file_list = []

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        name, _ = os.path.splitext(filename)
        file_list.append([{"name": name}, path])

    # Step 1: Run benchmarks
    print("\n🔬 Step 1: Running benchmarks...")
    benchmark = SpMVBenchmark()
    benchmark.benchmark_all(file_list)

    # Step 2: Save results
    print("\n💾 Step 2: Saving results...")
    benchmark.save_results()

    # Step 3: Visualize results
    print("\n📈 Step 3: Creating visualizations...")
    visualizer = ResultsVisualizer(benchmark.results)
    visualizer.print_summary()

    if HAS_PLOTTING:
        visualizer.create_performance_plots()
        print("✅ All visualizations saved in benchmark directory")
    else:
        print("⚠️ Plotting libraries not available, showing ASCII charts instead")
        visualizer.create_ascii_charts()

    # Final summary
    print("\n" + "="*80)
    print("🎉 BENCHMARK COMPLETE!")
    print("="*80)
    print(f"📁 All results saved in: {benchmark_dir.resolve()}")
    print("📄 Files generated:")
    print(f"   - benchmark_results.json (Raw data)")
    print(f"   - benchmark_summary_report.txt (Human-readable report)")
    if HAS_PLOTTING:
        print(f"   - spmv_benchmark_analysis.png (Comprehensive plots)")
        print(f"   - spmv_execution_times.png (Individual plot)")
        print(f"   - spmv_bandwidth.png (Individual plot)")
        print(f"   - spmv_bandwidth_vs_size.png (Individual plot)")
        print(f"   - spmv_speedup.png (Individual plot)")
        print(f"   - spmv_matrix_characteristics.png (Individual plot)")
        print(f"   - spmv_performance_vs_size.png (Individual plot)")
        print(f"   - spmv_efficiency.png (Individual plot)")
        print(f"   - spmv_detailed_matrix_analysis.png (Individual plot)")
        print("\n📊 Ready to analyze results with improved bandwidth calculations and connected plots!")
    print("="*80)

if __name__ == "__main__":
    main()
