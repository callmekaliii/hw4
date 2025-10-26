import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os


def extract_performance_data(filename):
    """Extract performance data from results files"""
    data = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    current_config = None

    for line in lines:
        line = line.strip()

        # Look for configuration lines
        config_match = re.search(
            r'threads=(\d+), memory=(\d+), iothreads=(\d+)', line)
        if config_match:
            current_config = {
                'threads': int(config_match.group(1)),
                'memory': int(config_match.group(2)),
                'iothreads': int(config_match.group(3))
            }

        # Look for final performance stats
        stats_match = re.search(
            r'vaultx t(\d+) i(\d+) m(\d+) k(\d+) ([0-9.]+) ([0-9.]+) ([0-9.]+)', line)
        if stats_match and current_config:
            t_threads, t_io, t_memory, t_k, hash_rate, io_rate, time = stats_match.groups()

            data.append({
                'threads': current_config['threads'],
                'memory': current_config['memory'],
                'iothreads': current_config['iothreads'],
                'k': int(t_k),
                'hash_rate': float(hash_rate),
                'io_rate': float(io_rate),
                'time': float(time)
            })

            current_config = None

    return pd.DataFrame(data)


def extract_real_performance(filename):
    """Extract real performance from progress monitor"""
    data = []

    with open(filename, 'r') as f:
        content = f.read()

    # Find all experiment sections
    experiments = re.findall(
        r'threads=(\d+), memory=(\d+), iothreads=(\d+)(.*?)(?=threads=\d|$)', content, re.DOTALL)

    for threads, memory, iothreads, experiment_text in experiments:
        threads = int(threads)
        memory = int(memory)
        iothreads = int(iothreads)

        # Determine K value
        if 'k=26' in filename.lower() or '67108864' in experiment_text:
            k = 26
            total_hashes = 67108864
        else:
            k = 32
            total_hashes = 4294967296

        # Look for hash rates in progress monitor
        hash_rates = re.findall(r'Hash Rate: ([0-9.]+) MH/s', experiment_text)
        io_rates = re.findall(r'I/O: ([0-9.]+) MB/s', experiment_text)

        if hash_rates:
            best_hash_rate = max(map(float, hash_rates))
            estimated_time = total_hashes / (best_hash_rate * 1e6)

            avg_io_rate = 0.0
            if io_rates:
                avg_io_rate = sum(map(float, io_rates)) / len(io_rates)

            data.append({
                'threads': threads,
                'memory': memory,
                'iothreads': iothreads,
                'k': k,
                'hash_rate': best_hash_rate,
                'io_rate': avg_io_rate,
                'time': estimated_time
            })

    return pd.DataFrame(data)


def analyze_search_results():
    """Analyze search experiment results"""
    if not os.path.exists('results_search.txt'):
        return pd.DataFrame()

    data = []

    with open('results_search.txt', 'r') as f:
        content = f.read()

    search_blocks = re.findall(
        r'k=(\d+), difficulty=(\d+)(.*?)(?=k=\d+,\s*difficulty=\d+|Search Summary:|$)', content, re.DOTALL)

    for k, difficulty, block in search_blocks:
        summary_match = re.search(
            r'Search Summary:.*?requested=(\d+).*?found_queries=(\d+).*?total_matches=(\d+).*?notfound=(\d+).*?total_time=([0-9.]+) s.*?avg_ms=([0-9.]+).*?searches/sec=([0-9.]+).*?total_seeks=(\d+).*?avg_seeks=([0-9.]+).*?total_comps=(\d+).*?avg_comps=([0-9.]+)', block, re.DOTALL)

        if summary_match:
            requested, found, matches, notfound, total_time, avg_ms, searches_sec, seeks, avg_seeks, comps, avg_comps = summary_match.groups()

            data.append({
                'K': int(k),
                'Difficulty': int(difficulty),
                'Searches': int(requested),
                'Found_Queries': int(found),
                'Total_Matches': int(matches),
                'Not_Found': int(notfound),
                'Total_Time': float(total_time),
                'Avg_Time_Per_Search_ms': float(avg_ms),
                'Searches_Per_Second': float(searches_sec),
                'Avg_Seeks_Per_Search': float(avg_seeks),
                'Avg_Comparisons_Per_Search': float(avg_comps),
                'Avg_Matches_Per_Found_Query': int(matches) / max(int(found), 1)
            })

    return pd.DataFrame(data)


def main():
    print("VAULTX COMPLETE PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Create output directory
    os.makedirs('results_analysis', exist_ok=True)

    # Extract performance data
    print("Extracting performance data...")
    df_k26_final = extract_performance_data('results_small.txt')
    df_k32_final = extract_performance_data('results_large.txt')
    df_k26_real = extract_real_performance('results_small.txt')

    # Use real performance for K=26, final stats for K-32
    df_k26 = df_k26_real if len(df_k26_real) > 0 else df_k26_final
    df_k32 = df_k32_final

    print(f"K=26 experiments: {len(df_k26)}")
    print(f"K=32 experiments: {len(df_k32)}")

    # Analyze search results
    df_search = analyze_search_results()
    if len(df_search) > 0:
        print(f"Search experiments: {len(df_search)}")

    # Generate K=26 tables
    print("\n" + "="*50)
    print("K=26 PERFORMANCE TABLES")
    print("="*50)

    for io_threads in [1, 2, 4]:
        df_io = df_k26[df_k26['iothreads'] == io_threads]

        print(f"\n--- I/O Threads = {io_threads} ---")

        # Time table
        time_table = df_io.pivot_table(
            index='threads',
            columns='memory',
            values='time',
            aggfunc='first'
        ).round(1)

        print("Execution Time (seconds):")
        print(time_table)
        time_table.to_csv(f'results_analysis/k26_time_i{io_threads}.csv')

        # Hash rate table
        hash_table = df_io.pivot_table(
            index='threads',
            columns='memory',
            values='hash_rate',
            aggfunc='first'
        ).round(2)

        print("\nHash Rate (MH/s):")
        print(hash_table)
        hash_table.to_csv(f'results_analysis/k26_hashrate_i{io_threads}.csv')

    # Generate K-32 table
    print("\n" + "="*50)
    print("K=32 PERFORMANCE TABLE")
    print("="*50)

    time_table_k32 = df_k32.pivot_table(
        index='threads',
        columns='memory',
        values='time',
        aggfunc='first'
    ).round(1)

    print("Execution Time (seconds):")
    print(time_table_k32)
    time_table_k32.to_csv('results_analysis/k32_time.csv')

    hash_table_k32 = df_k32.pivot_table(
        index='threads',
        columns='memory',
        values='hash_rate',
        aggfunc='first'
    ).round(3)

    print("\nHash Rate (MH/s):")
    print(hash_table_k32)
    hash_table_k32.to_csv('results_analysis/k32_hashrate.csv')

    # Generate search table
    if len(df_search) > 0:
        print("\n" + "="*50)
        print("SEARCH PERFORMANCE TABLE")
        print("="*50)

        search_table = df_search[['K', 'Difficulty', 'Searches', 'Avg_Seeks_Per_Search',
                                 'Total_Time', 'Avg_Time_Per_Search_ms', 'Searches_Per_Second',
                                  'Found_Queries', 'Not_Found']]
        print(search_table.round(2))
        search_table.to_csv(
            'results_analysis/search_performance.csv', index=False)

    # Generate plots
    print("\nGenerating performance plots...")
    generate_plots(df_k26, df_k32)

    # Performance summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)

    # K=26 best configuration
    best_k26 = df_k26.loc[df_k26['hash_rate'].idxmax()]
    print(f"Best K=26 Configuration:")
    print(
        f"  Threads: {best_k26['threads']}, Memory: {best_k26['memory']}MB, I/O: {best_k26['iothreads']}")
    print(
        f"  Hash Rate: {best_k26['hash_rate']:.2f} MH/s, Time: {best_k26['time']:.1f}s")

    # K=32 best configuration
    best_k32 = df_k32.loc[df_k32['hash_rate'].idxmax()]
    print(f"\nBest K=32 Configuration:")
    print(f"  Memory: {best_k32['memory']}MB")
    print(
        f"  Hash Rate: {best_k32['hash_rate']:.3f} MH/s, Time: {best_k32['time']:.1f}s")

    # Scaling analysis
    print(f"\nScaling Analysis:")
    df_scaling = df_k26[df_k26['memory'] == 1024]
    single_thread = df_scaling[df_scaling['threads'] == 1]['hash_rate'].mean()
    multi_thread = df_scaling[df_scaling['threads'] == 48]['hash_rate'].mean()
    print(f"  Thread scaling (1→48): {multi_thread/single_thread:.1f}x")

    print(f"\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)
    print("Generated in 'results_analysis' directory:")
    print("  - K=26 performance tables for I/O threads 1, 2, 4")
    print("  - K=32 performance table")
    if len(df_search) > 0:
        print("  - Search performance table")
    print("  - Performance plots")
    print("  - Best configuration recommendations")


def generate_plots(df_k26, df_k32):
    """Generate performance plots"""

    # K=26 plots
    for io_threads in [1, 2, 4]:
        df_io = df_k26[df_k26['iothreads'] == io_threads]

        plt.figure(figsize=(10, 6))
        for memory in [256, 512, 1024]:
            df_mem = df_io[df_io['memory'] == memory].sort_values('threads')
            plt.plot(df_mem['threads'], df_mem['hash_rate'],
                     marker='o', linewidth=2, markersize=6, label=f'{memory} MB')

        plt.title(
            f'K=26: Hash Rate vs Compute Threads (I/O Threads = {io_threads})')
        plt.xlabel('Compute Threads')
        plt.ylabel('Hash Rate (MH/s)')
        plt.legend(title='Memory Size')
        plt.grid(True, alpha=0.3)
        plt.xticks([1, 2, 4, 8, 12, 24, 48])
        plt.tight_layout()
        plt.savefig(
            f'results_analysis/k26_i{io_threads}_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    # K=32 plots
    if len(df_k32) > 0:
        df_sorted = df_k32.sort_values('memory')

        plt.figure(figsize=(10, 6))
        plt.plot(df_sorted['memory'], df_sorted['time'],
                 marker='s', linewidth=2, markersize=8, color='red')
        plt.title('K=32: Execution Time vs Memory Size (Threads=24, I/O Threads=1)')
        plt.xlabel('Memory Size (MB)')
        plt.ylabel('Execution Time (seconds)')
        plt.grid(True, alpha=0.3)
        plt.xticks(df_sorted['memory'])
        plt.tight_layout()
        plt.savefig('results_analysis/k32_time_vs_memory.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(df_sorted['memory'], df_sorted['hash_rate'],
                 marker='o', linewidth=2, markersize=8, color='blue')
        plt.title('K=32: Hash Rate vs Memory Size (Threads=24, I/O Threads=1)')
        plt.xlabel('Memory Size (MB)')
        plt.ylabel('Hash Rate (MH/s)')
        plt.grid(True, alpha=0.3)
        plt.xticks(df_sorted['memory'])
        plt.tight_layout()
        plt.savefig('results_analysis/k32_hashrate_vs_memory.png',
                    dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    main()
