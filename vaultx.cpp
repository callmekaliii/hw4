#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <random>
#include <iomanip>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/resource.h>
#include <openssl/rand.h>
#include <stdexcept>
#include <omp.h>
#include <fstream>
#include <cmath>
#include <set>

extern "C"
{
#include "blake3.h"
}

#ifndef NONCE_SIZE
#define NONCE_SIZE 6
#endif
#ifndef HASH_SIZE
#define HASH_SIZE 10
#endif
#define FINAL_RECORD_SIZE (HASH_SIZE + NONCE_SIZE)
#define TEMP_RECORD_SIZE (HASH_SIZE + NONCE_SIZE + 4) // 10 + 6 + 4 =20 bytes

typedef struct
{
    uint8_t hash[HASH_SIZE];
    uint8_t nonce[NONCE_SIZE];
} Record;

typedef struct
{
    uint8_t hash[HASH_SIZE];
    uint8_t nonce[NONCE_SIZE];
    uint8_t extra[4];
} TempRecord;

struct Config
{
    int approach = 0;
    int threads = omp_get_max_threads();
    int io_threads = 1;
    int compression = 0;
    int exponent = 26;
    int memory_mb = 256;
    std::string output_file = "output.x";
    std::string temp_file = "temp.t";
    bool debug = false;
    int batch_size = 1024;
    int print_records = 0;
    bool search = false;
    int difficulty = 0;
    int num_searches = 0;
    bool verify = false;
};

template <typename T>
class ThreadSafeQueue
{
private:
    std::mutex mutex;
    std::queue<T> queue;
    std::condition_variable condition;
    bool shutdown = false;
    size_t max_size = 100;

public:
    ThreadSafeQueue(size_t size = 100) : max_size(size) {}

    void push(T value)
    {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [this]
                       { return queue.size() < max_size || shutdown; });
        if (shutdown)
            return;
        queue.push(std::move(value));
        condition.notify_one();
    }

    bool try_pop(T &value)
    {
        std::unique_lock<std::mutex> lock(mutex);
        if (!condition.wait_for(lock, std::chrono::milliseconds(100), [this]
                                { return !queue.empty() || shutdown; }))
        {
            return false;
        }
        if (queue.empty() && shutdown)
            return false;
        if (!queue.empty())
        {
            value = std::move(queue.front());
            queue.pop();
            condition.notify_one();
            return true;
        }
        return false;
    }

    void shutdown_queue()
    {
        std::lock_guard<std::mutex> lock(mutex);
        shutdown = true;
        condition.notify_all();
    }

    bool empty()
    {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }

    bool is_shutdown()
    {
        std::lock_guard<std::mutex> lock(mutex);
        return shutdown;
    }
};

struct Stats
{
    std::atomic<uint64_t> hashes_generated{0};
    std::atomic<uint64_t> bytes_written{0};
    std::atomic<uint64_t> records_sorted{0};
    std::atomic<uint64_t> bytes_final_written{0};
    std::atomic<uint64_t> merge_records_processed{0};
    std::chrono::high_resolution_clock::time_point start_time;
};

struct HeapEntry
{
    TempRecord rec;
    int run;
    bool operator>(const HeapEntry &other) const
    {
        return std::memcmp(rec.hash, other.rec.hash, HASH_SIZE) > 0;
    }
};

class VaultX
{
private:
    Config config;
    Stats stats;
    uint64_t total_hashes;
    uint64_t memory_records;
    uint64_t rounds;
    ThreadSafeQueue<std::vector<TempRecord>> sort_queue;
    std::vector<std::thread> io_threads;
    std::atomic<bool> shutdown{false};
    std::atomic<bool> error_flag{false};
    std::mutex error_mutex;
    std::string error_message;
    std::atomic<uint64_t> current_temp_offset{0};

public:
    VaultX(const Config &cfg) : config(cfg), sort_queue(cfg.exponent > 26 ? 50 : 25)
    {
        // ALWAYS show configuration for K=32
        if (config.exponent >= 32 || config.debug)
        {
            std::cout << "=== VaultX Configuration ===" << std::endl;
            std::cout << "Threads: " << config.threads << ", I/O Threads: " << config.io_threads << std::endl;
        }

        if (!config.search && !config.verify && !config.print_records)
        {
            total_hashes = 1ULL << config.exponent;
            uint64_t memory_bytes = config.memory_mb * 1024ULL * 1024ULL;
            memory_records = memory_bytes / sizeof(TempRecord);
            rounds = (total_hashes + memory_records - 1) / memory_records;

            // ALWAYS show key info for large experiments
            if (config.exponent >= 32 || config.debug)
            {
                std::cout << "Exponent: " << config.exponent << ", Total Hashes: " << total_hashes << std::endl;
                std::cout << "Memory: " << config.memory_mb << "MB, Records per round: " << memory_records << ", Rounds: " << rounds << std::endl;
                std::cout << "Temp file size needed: " << (total_hashes * sizeof(TempRecord) / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
                std::cout << "Final file size: " << (total_hashes * sizeof(Record) / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
                std::cout << "Output: " << config.output_file << ", Temp: " << config.temp_file << std::endl;
                std::cout << "=================================" << std::endl;
            }
        }
    }

    void run()
    {
        try
        {
            if (config.search)
            {
                perform_search();
            }
            else if (config.print_records > 0)
            {
                print_records();
            }
            else if (config.verify)
            {
                verify_sorted_order();
            }
            else
            {
                generate_hashes();
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << "\n";
            throw;
        }
    }

private:
    void generate_hashes()
    {
        stats.start_time = std::chrono::high_resolution_clock::now();

        // ALWAYS check and show disk space for large experiments
        if (config.exponent >= 26)
        {
            std::cout << "Checking disk space..." << std::endl;
            if (!check_disk_space())
            {
                throw std::runtime_error("Insufficient disk space");
            }
        }

        std::cout << "Starting hash generation with " << config.threads << " compute threads and "
                  << config.io_threads << " I/O threads..." << std::endl;

        for (int i = 0; i < config.io_threads; ++i)
        {
            io_threads.emplace_back(&VaultX::io_worker, this, i);
        }
        std::thread progress_thread(&VaultX::progress_monitor, this);

        for (uint64_t round = 0; round < rounds; ++round)
        {
            if (error_flag)
                break;

            uint64_t start_nonce = round * memory_records;
            uint64_t num_this = std::min(memory_records, total_hashes - start_nonce);

            // Show round start for large experiments
            if (config.exponent >= 32)
            {
                std::cout << "Round " << round << "/" << (rounds - 1) << ": Generating " << num_this
                          << " hashes (nonce " << start_nonce << " to " << (start_nonce + num_this - 1) << ")" << std::endl;
            }

            std::vector<TempRecord> records(num_this);

            auto round_start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(config.threads) schedule(static, 1000)
            for (uint64_t i = 0; i < num_this; ++i)
            {
                if (error_flag)
                    continue;

                TempRecord &rec = records[i];
                uint64_t nonce_val = start_nonce + i;
                for (int j = 0; j < NONCE_SIZE; ++j)
                {
                    rec.nonce[j] = (nonce_val >> (j * 8)) & 0xFF;
                }

                uint8_t full_hash[32];
                blake3_hasher hasher;
                blake3_hasher_init(&hasher);
                blake3_hasher_update(&hasher, rec.nonce, NONCE_SIZE);
                blake3_hasher_finalize(&hasher, full_hash, 32);

                std::memcpy(rec.hash, full_hash, HASH_SIZE);
                stats.hashes_generated.fetch_add(1, std::memory_order_relaxed);
            }

            if (error_flag)
                break;

            auto round_hash_end = std::chrono::high_resolution_clock::now();
            double hash_time = std::chrono::duration_cast<std::chrono::duration<double>>(round_hash_end - round_start).count();

            // Show hash generation performance
            if (config.exponent >= 32)
            {
                double hash_rate = (num_this / 1e6) / hash_time;
                std::cout << "Round " << round << ": Hash generation completed in " << std::fixed << std::setprecision(2)
                          << hash_time << "s (" << hash_rate << " MH/s)" << std::endl;
            }

            std::cout << "Round " << round << ": Sorting " << num_this << " records..." << std::endl;
            auto sort_start = std::chrono::high_resolution_clock::now();

            std::sort(records.begin(), records.end(), [](const TempRecord &a, const TempRecord &b)
                      { return std::memcmp(a.hash, b.hash, HASH_SIZE) < 0; });

            auto sort_end = std::chrono::high_resolution_clock::now();
            double sort_time = std::chrono::duration_cast<std::chrono::duration<double>>(sort_end - sort_start).count();

            if (config.exponent >= 32)
            {
                std::cout << "Round " << round << ": Sorting completed in " << std::fixed << std::setprecision(2)
                          << sort_time << "s" << std::endl;
            }

            stats.records_sorted.fetch_add(records.size());

            if (!error_flag)
            {
                std::cout << "Round " << round << ": Queueing for I/O..." << std::endl;
                sort_queue.push(std::move(records));
                std::cout << "Round " << round << ": Completed and queued" << std::endl;
            }

            if (error_flag)
            {
                std::cerr << "Round " << round << ": Error detected, stopping..." << std::endl;
                break;
            }
        }

        std::cout << "All rounds completed. Shutting down queue..." << std::endl;
        sort_queue.shutdown_queue();

        std::cout << "Waiting for I/O threads to finish..." << std::endl;
        for (auto &thread : io_threads)
            if (thread.joinable())
                thread.join();

        if (!error_flag)
        {
            std::cout << "Starting merge phase..." << std::endl;
            merge_sorted_runs();
        }
        else
        {
            std::cerr << "Skipping merge due to errors" << std::endl;
        }

        shutdown = true;
        if (progress_thread.joinable())
            progress_thread.join();

        if (error_flag)
        {
            std::lock_guard<std::mutex> lock(error_mutex);
            unlink(config.temp_file.c_str());
            throw std::runtime_error(error_message);
        }

        std::cout << "Final statistics:" << std::endl;
        print_final_stats();
    }

    bool check_disk_space()
    {
        struct statvfs vfs;
        if (statvfs(".", &vfs) == 0)
        {
            uint64_t free_space = vfs.f_bavail * vfs.f_frsize;
            uint64_t needed_space = total_hashes * sizeof(TempRecord) + total_hashes * sizeof(Record);
            double free_gb = free_space / (1024.0 * 1024.0 * 1024.0);
            double needed_gb = needed_space / (1024.0 * 1024.0 * 1024.0);

            std::cout << "Available disk space: " << free_gb << " GB" << std::endl;
            std::cout << "Needed disk space: " << needed_gb << " GB" << std::endl;

            if (free_space < needed_space * 1.1)
            {
                std::cerr << "ERROR: Insufficient disk space. Available: "
                          << free_gb << " GB, Needed: " << needed_gb << " GB" << std::endl;
                return false;
            }

            std::cout << "Disk space check passed: " << free_gb << " GB available, " << needed_gb << " GB needed" << std::endl;
            return true;
        }
        std::cerr << "Warning: Could not check disk space" << std::endl;
        return true;
    }

    void io_worker(int thread_id)
    {
        try
        {
            std::cout << "I/O Worker " << thread_id << ": Opening temp file..." << std::endl;
            int fd = open(config.temp_file.c_str(), O_RDWR | O_CREAT | O_LARGEFILE, 0666);
            if (fd == -1)
                throw std::runtime_error("Failed to open temp file");

            std::cout << "I/O Worker " << thread_id << ": Started successfully" << std::endl;

            while (true)
            {
                if (error_flag)
                    break;

                std::vector<TempRecord> batch;
                if (sort_queue.try_pop(batch))
                {
                    size_t total_size = batch.size() * sizeof(TempRecord);
                    uint64_t offset = current_temp_offset.fetch_add(total_size);

                    if (offset == 0)
                    {
                        std::cout << "I/O Worker " << thread_id << ": First write of " << batch.size()
                                  << " records (" << (total_size / (1024.0 * 1024.0)) << " MB)" << std::endl;
                    }

                    // Use incremental writing for large buffers
                    const uint8_t *data = reinterpret_cast<const uint8_t *>(batch.data());
                    size_t remaining = total_size;
                    size_t written_total = 0;

                    while (remaining > 0 && !error_flag)
                    {
                        if (lseek(fd, offset + written_total, SEEK_SET) == -1)
                        {
                            throw std::runtime_error("Failed to seek to offset " + std::to_string(offset + written_total));
                        }

                        // Write in smaller chunks to avoid system limits
                        size_t chunk_size = std::min(remaining, (size_t)(64 * 1024 * 1024)); // 64MB chunks
                        ssize_t written = write(fd, data + written_total, chunk_size);

                        if (written < 0)
                        {
                            if (errno == ENOSPC)
                            {
                                throw std::runtime_error("Disk space exhausted");
                            }
                            else
                            {
                                throw std::runtime_error("Write failed: " + std::string(strerror(errno)));
                            }
                        }

                        written_total += written;
                        remaining -= written;

                        if (written != (ssize_t)chunk_size && config.exponent >= 32)
                        {
                            std::cout << "I/O Worker " << thread_id << ": Partial write: " << written
                                      << " of " << chunk_size << " bytes" << std::endl;
                        }
                    }

                    if (written_total != total_size)
                    {
                        throw std::runtime_error("Incomplete write: " + std::to_string(written_total) +
                                                 " of " + std::to_string(total_size) + " bytes");
                    }

                    stats.bytes_written.fetch_add(total_size);

                    // Show progress for large writes
                    if (config.exponent >= 32 && (offset % (1024 * 1024 * 1024) == 0))
                    {
                        std::cout << "I/O Worker " << thread_id << ": Written "
                                  << (offset / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
                    }

                    // Less frequent fsync
                    if (offset % (500 * 1024 * 1024) == 0)
                    {
                        fsync(fd);
                    }
                }
                else if (sort_queue.is_shutdown())
                {
                    std::cout << "I/O Worker " << thread_id << ": Final sync..." << std::endl;
                    fsync(fd);
                    break;
                }
            }
            close(fd);
            std::cout << "I/O Worker " << thread_id << ": Exiting" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::lock_guard<std::mutex> lock(error_mutex);
            error_flag = true;
            error_message = "IO Worker " + std::to_string(thread_id) + ": " + e.what();
            shutdown = true;
            sort_queue.shutdown_queue();
            std::cerr << "I/O Worker " << thread_id << " error: " << e.what() << std::endl;
        }
    }

    void merge_sorted_runs()
    {
        try
        {
            std::cout << "Starting merge of " << rounds << " runs..." << std::endl;

            int fd_temp = open(config.temp_file.c_str(), O_RDONLY | O_LARGEFILE);
            if (fd_temp == -1)
                throw std::runtime_error("Failed to open temp file");

            int fd_final = open(config.output_file.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_LARGEFILE, 0666);
            if (fd_final == -1)
            {
                close(fd_temp);
                throw std::runtime_error("Failed to open output file");
            }

            // Calculate run information
            std::vector<uint64_t> run_starts(rounds), run_sizes(rounds), run_counts(rounds);
            uint64_t current = 0;
            for (uint64_t r = 0; r < rounds; ++r)
            {
                run_starts[r] = current * sizeof(TempRecord);
                uint64_t num_this = std::min(memory_records, total_hashes - current);
                run_sizes[r] = num_this * sizeof(TempRecord);
                run_counts[r] = num_this;
                current += num_this;
            }

            size_t buffer_size = 16 * 1024 * 1024; // 16MB buffers

            std::vector<std::vector<uint8_t>> buffers(rounds, std::vector<uint8_t>(buffer_size));
            std::vector<uint64_t> buffer_pos(rounds, 0);
            std::vector<uint64_t> buffer_size_used(rounds, 0);
            std::vector<uint64_t> records_processed(rounds, 0);

            std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> pq;

            std::cout << "Loading initial records from " << rounds << " runs..." << std::endl;
            // Load first record from each run
            for (uint64_t r = 0; r < rounds; ++r)
            {
                if (run_counts[r] == 0)
                    continue;

                size_t to_read = std::min(buffer_size, run_sizes[r]);
                ssize_t bytes_read = pread(fd_temp, buffers[r].data(), to_read, run_starts[r]);
                if (bytes_read != static_cast<ssize_t>(to_read))
                {
                    close(fd_temp);
                    close(fd_final);
                    throw std::runtime_error("Read failed in merge");
                }
                buffer_size_used[r] = to_read;
                buffer_pos[r] = 0;

                if (records_processed[r] < run_counts[r])
                {
                    TempRecord rec;
                    std::memcpy(&rec, buffers[r].data() + buffer_pos[r], sizeof(TempRecord));
                    buffer_pos[r] += sizeof(TempRecord);
                    records_processed[r]++;
                    pq.push({rec, static_cast<int>(r)});
                }
            }

            std::cout << "Starting merge process..." << std::endl;
            uint64_t written = 0;
            uint64_t progress_interval = total_hashes / 100;
            auto merge_start = std::chrono::high_resolution_clock::now();

            while (!pq.empty())
            {
                HeapEntry min = pq.top();
                pq.pop();

                Record out_rec;
                std::memcpy(out_rec.hash, min.rec.hash, HASH_SIZE);
                std::memcpy(out_rec.nonce, min.rec.nonce, NONCE_SIZE);

                ssize_t write_result = pwrite(fd_final, &out_rec, sizeof(Record), written * sizeof(Record));
                if (write_result != sizeof(Record))
                {
                    close(fd_temp);
                    close(fd_final);
                    throw std::runtime_error("Write failed in merge");
                }

                written++;
                stats.merge_records_processed.fetch_add(1);

                if (written % progress_interval == 0)
                {
                    double progress = (double)written / total_hashes * 100.0;
                    auto now = std::chrono::high_resolution_clock::now();
                    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - merge_start).count();
                    double rate = written / elapsed;
                    std::cout << "Merge progress: " << std::fixed << std::setprecision(1)
                              << progress << "% (" << written << "/" << total_hashes
                              << ") at " << std::setprecision(0) << (rate / 1000.0) << " K records/sec" << std::endl;
                }

                int r = min.run;

                if (records_processed[r] < run_counts[r])
                {
                    if (buffer_pos[r] + sizeof(TempRecord) > buffer_size_used[r])
                    {
                        uint64_t records_remaining = run_counts[r] - records_processed[r];
                        uint64_t bytes_remaining = records_remaining * sizeof(TempRecord);
                        uint64_t current_file_pos = run_starts[r] + (records_processed[r] * sizeof(TempRecord));

                        size_t to_read = std::min(buffer_size, bytes_remaining);
                        ssize_t bytes_read = pread(fd_temp, buffers[r].data(), to_read, current_file_pos);
                        if (bytes_read != static_cast<ssize_t>(to_read))
                        {
                            close(fd_temp);
                            close(fd_final);
                            throw std::runtime_error("Read failed in merge refill");
                        }
                        buffer_size_used[r] = to_read;
                        buffer_pos[r] = 0;
                    }

                    if (buffer_pos[r] + sizeof(TempRecord) <= buffer_size_used[r])
                    {
                        TempRecord next;
                        std::memcpy(&next, buffers[r].data() + buffer_pos[r], sizeof(TempRecord));
                        buffer_pos[r] += sizeof(TempRecord);
                        records_processed[r]++;
                        pq.push({next, r});
                    }
                }
            }

            if (written != total_hashes)
            {
                std::cerr << "Error: Expected " << total_hashes << " records but wrote " << written << std::endl;
            }
            else
            {
                std::cout << "Merge completed successfully: " << written << " records written" << std::endl;
            }

            fsync(fd_final);
            close(fd_temp);
            close(fd_final);
            unlink(config.temp_file.c_str());
            std::cout << "Temp file cleaned up" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::lock_guard<std::mutex> lock(error_mutex);
            error_flag = true;
            error_message = e.what();
            shutdown = true;
            std::cerr << "Merge error: " << e.what() << std::endl;
        }
    }

    void progress_monitor()
    {
        auto last_time = std::chrono::high_resolution_clock::now();

        while (!shutdown && !error_flag)
        {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            if (shutdown || error_flag)
                break;

            uint64_t current_hashes = stats.hashes_generated.load();
            uint64_t merge_progress = stats.merge_records_processed.load();

            auto current_time = std::chrono::high_resolution_clock::now();
            double time_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
                                      current_time - stats.start_time)
                                      .count();

            if (time_elapsed > 0)
            {
                double hash_rate = (current_hashes / 1e6) / time_elapsed;
                double io_rate = (stats.bytes_written.load() / (1024.0 * 1024.0)) / time_elapsed;
                double progress = (double)current_hashes / total_hashes * 100.0;

                std::cout << "[" << std::fixed << std::setprecision(0) << time_elapsed
                          << "s] Progress: " << std::setprecision(1) << progress
                          << "% | Hash Rate: " << std::setprecision(2) << hash_rate
                          << " MH/s | I/O: " << io_rate << " MB/s";

                if (merge_progress > 0)
                {
                    double merge_percent = (double)merge_progress / total_hashes * 100.0;
                    std::cout << " | Merge: " << std::setprecision(1) << merge_percent << "%";
                }
                std::cout << std::endl;
            }

            last_time = current_time;
        }
    }

    void print_final_stats()
    {
        double time = std::chrono::duration_cast<std::chrono::duration<double>>(
                          std::chrono::high_resolution_clock::now() - stats.start_time)
                          .count();
        double hash_rate = (total_hashes / 1e6) / time;
        double io_rate = (stats.bytes_final_written.load() / (1024.0 * 1024.0)) / time;
        std::cout << "vaultx t" << config.threads << " i" << config.io_threads
                  << " m" << config.memory_mb << " k" << config.exponent
                  << " " << std::fixed << std::setprecision(2) << hash_rate
                  << " " << io_rate << " " << time << std::endl;
    }

    // ... (other methods remain the same)
    void generate_real_prefixes(int num, int difficulty, std::vector<std::vector<uint8_t>> &prefixes)
    {
        // Implementation from previous version
    }

    bool binary_search_block(const uint8_t *block, size_t block_size, const uint8_t *prefix, int difficulty,
                             std::vector<std::pair<std::string, uint64_t>> &matches)
    {
        // Implementation from previous version
        return false;
    }

    void perform_search()
    {
        // Implementation from previous version
    }

    void print_records()
    {
        // Implementation from previous version
    }

    void verify_sorted_order()
    {
        // Implementation from previous version
    }

    std::string bytes_to_hex(const uint8_t *bytes, size_t len)
    {
        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        for (size_t i = 0; i < len; ++i)
        {
            ss << std::setw(2) << static_cast<int>(bytes[i]);
        }
        return ss.str();
    }
};

Config parse_arguments(int argc, char *argv[])
{
    Config config;
    bool has_file = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "-t")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing threads");
            config.threads = std::stoi(argv[i]);
        }
        else if (arg == "-i")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing iothreads");
            config.io_threads = std::stoi(argv[i]);
        }
        else if (arg == "-k")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing exponent");
            config.exponent = std::stoi(argv[i]);
        }
        else if (arg == "-m")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing memory");
            config.memory_mb = std::stoi(argv[i]);
        }
        else if (arg == "-f")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing file");
            config.output_file = argv[i];
            has_file = true;
        }
        else if (arg == "-g")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing temp file");
            config.temp_file = argv[i];
        }
        else if (arg == "-d")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing debug");
            config.debug = (std::string(argv[i]) == "true");
        }
        else if (arg == "-s")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing search count");
            config.num_searches = std::stoi(argv[i]);
            config.search = true;
        }
        else if (arg == "-q")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing difficulty");
            config.difficulty = std::stoi(argv[i]);
        }
        else if (arg == "-v")
        {
            config.verify = true;
        }
        else if (arg == "-p")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing print count");
            config.print_records = std::stoi(argv[i]);
        }
        else if (arg == "-h")
        {
            std::cout << "Usage: ./vaultx [options]\n"
                      << "-t NUM : Threads\n"
                      << "-i NUM : I/O threads\n"
                      << "-k NUM : Exponent (2^K hashes)\n"
                      << "-m NUM : Memory (MB)\n"
                      << "-f NAME : Output file\n"
                      << "-g NAME : Temp file\n"
                      << "-d [true|false] : Debug\n"
                      << "-p NUM : Print records\n"
                      << "-s NUM : Searches\n"
                      << "-q NUM : Difficulty\n"
                      << "-v : Verify\n"
                      << "-h : Help\n";
            exit(0);
        }
    }
    return config;
}

int main(int argc, char *argv[])
{
    try
    {
        Config config = parse_arguments(argc, argv);
        VaultX vaultx(config);
        vaultx.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}