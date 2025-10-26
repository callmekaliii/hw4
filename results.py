import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os


def parse_results(file):
    data = []
    with open(file, 'r') as f:
        lines = f.readlines()

    current_config = {}
    for line in lines:
        # Look for configuration lines
        config_match = re.search(
            r'threads=(\d+), memory=(\d+), iothreads=(\d+)', line)
        if config_match:
            current_config = {
                'threads': int(config_match.group(1)),
                'memory': int(config_match.group(2)),
                'iothreads': int(config_match.group(3))
            }

        # Look for performance output
        perf_match = re.search(
            r'vaultx t(\d+) i(\d+) m(\d+) k(\d+) ([0-9.]+) ([0-9.]+) ([0-9.]+)', line)
        if perf_match and current_config:
            threads, iothreads, memory, k, hash_rate, io_rate, time = map(
                float, perf_match.groups())
            data.append({
                'threads': int(threads),
                'iothreads': int(iothreads),
                'memory': int(memory),
                'k': int(k),
                'hash_rate': hash_rate,
                'io_rate': io_rate,
                'time': time
            })
            current_config = {}  # Reset for next entry

    return pd.DataFrame(data)


def parse_search(file):
    data = []
    with open(file, 'r') as f:
        content = f.read()

    # Find all search summary blocks
    summaries = re.findall(
        r'k=(\d+), difficulty=(\d+).*?Search Summary: requested=(\d+) found_queries=(\d+) total_matches=(\d+) notfound=(\d+) total_time=([0-9.]+) s avg_ms=([0-9.]+) searches/sec=([0-9.]+) total_seeks=(\d+) avg_seeks=([0-9.]+) total_comps=(\d+) avg_comps=([0-9.]+) avg_matches_per_found=([0-9.]+)',
        content, re.DOTALL
    )

    for summary in summaries:
        k, difficulty, requested, found, matches, notfound, total_time, avg_ms, searches_sec, seeks, avg_seeks, comps, avg_comps, avg_matches = summary
        data.append({
            'k': int(k),
            'difficulty': int(difficulty),
            'searches': int(requested),
            'found': int(found),
            'matches': int(matches),
            'notfound': int(notfound),
            'time': float(total_time),
            'avg_ms': float(avg_ms),
            'searches_per_sec': float(searches_sec),
            'avg_seeks': float(avg_seeks),
            'avg_comps': float(avg_comps),
            'avg_matches': float(avg_matches)
        })

    return pd.DataFrame(data)


def generate_plots():
    # Create plots directory
    os.makedirs('plots', exist_ok=True)

    # Parse results
    try:
        df_small = parse_results('results_small.txt')
        df_large = parse_results('results_large.txt')
        df_search = parse_search('results_search.txt')

        print(f"Small experiments: {len(df_small)} records")
        print(f"Large experiments: {len(df_large)} records")
        print(f"Search experiments: {len(df_search)} records")

        # Plot 1: K=26, Fixed I/O threads = 1
        if not df_small.empty:
            plt.figure(figsize=(12, 8))
            df_i1 = df_small[df_small['iothreads'] == 1]
            sns.lineplot(data=df_i1, x='threads', y='hash_rate',
                         hue='memory', marker='o', linewidth=2.5)
            plt.title('Hash Rate vs Threads (K=26, I/O Threads=1)', fontsize=14)
            plt.xlabel('Compute Threads', fontsize=12)
            plt.ylabel('Hash Rate (MH/s)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig('plots/hash_rate_k26_i1.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

            # Plot 2: K=26, Fixed I/O threads = 2
            plt.figure(figsize=(12, 8))
            df_i2 = df_small[df_small['iothreads'] == 2]
            sns.lineplot(data=df_i2, x='threads', y='hash_rate',
                         hue='memory', marker='s', linewidth=2.5)
            plt.title('Hash Rate vs Threads (K=26, I/O Threads=2)', fontsize=14)
            plt.xlabel('Compute Threads', fontsize=12)
            plt.ylabel('Hash Rate (MH/s)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig('plots/hash_rate_k26_i2.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

            # Plot 3: K=26, Fixed I/O threads = 4
            plt.figure(figsize=(12, 8))
            df_i4 = df_small[df_small['iothreads'] == 4]
            sns.lineplot(data=df_i4, x='threads', y='hash_rate',
                         hue='memory', marker='^', linewidth=2.5)
            plt.title('Hash Rate vs Threads (K=26, I/O Threads=4)', fontsize=14)
            plt.xlabel('Compute Threads', fontsize=12)
            plt.ylabel('Hash Rate (MH/s)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig('plots/hash_rate_k26_i4.png',
                        dpi=300, bbox_inches='tight')
            plt.close()

        # Plot 4: K=32, Time vs Memory
        if not df_large.empty:
            plt.figure(figsize=(12, 8))
            sns.lineplot(data=df_large, x='memory',
                         y='time', marker='o', linewidth=2.5)
            plt.title(
                'Time vs Memory (K=32, Threads=24, I/O Threads=1)', fontsize=14)
            plt.xlabel('Memory (MB)', fontsize=12)
            plt.ylabel('Time (s)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig('plots/time_k32_t24.png', dpi=300, bbox_inches='tight')
            plt.close()

        # Generate tables
        def generate_table(df, k, iothreads=None):
            if iothreads is None:
                pivot = df.pivot(
                    index='threads', columns='memory', values='time')
            else:
                pivot = df[df['iothreads'] == iothreads].pivot(
                    index='threads', columns='memory', values='time')
            return pivot

        # Save tables
        if not df_small.empty:
            table_k26_i1 = generate_table(df_small, k=26, iothreads=1)
            table_k26_i2 = generate_table(df_small, k=26, iothreads=2)
            table_k26_i4 = generate_table(df_small, k=26, iothreads=4)

            table_k26_i1.to_csv('plots/table_k26_i1.csv')
            table_k26_i2.to_csv('plots/table_k26_i2.csv')
            table_k26_i4.to_csv('plots/table_k26_i4.csv')

        if not df_large.empty:
            table_k32 = generate_table(df_large, k=32)
            table_k32.to_csv('plots/table_k32.csv')

        if not df_search.empty:
            search_table = df_search[['k', 'difficulty', 'searches', 'avg_seeks',
                                     'time', 'avg_ms', 'searches_per_sec', 'found', 'notfound']]
            search_table.to_csv('plots/table_search.csv')

        print("Analysis completed successfully!")
        print("Check the 'plots' directory for generated charts and tables.")

    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    generate_plots()
