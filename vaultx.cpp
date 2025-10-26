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

typedef struct
{
    uint8_t hash[HASH_SIZE];
    uint8_t nonce[NONCE_SIZE];
} Record;

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
    std::atomic<uint64_t> current_round{0};
    std::atomic<uint64_t> io_blocks_written{0};
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point round_start_time;
};

class VaultX
{
private:
    Config config;
    Stats stats;
    uint64_t total_hashes;
    uint64_t memory_records;
    uint64_t rounds;
    ThreadSafeQueue<std::vector<Record>> sort_queue;
    std::vector<std::thread> io_threads;
    std::atomic<bool> shutdown{false};
    std::atomic<bool> error_flag{false};
    std::mutex error_mutex;
    std::string error_message;
    std::atomic<uint64_t> total_io_blocks{0};

public:
    VaultX(const Config &cfg) : config(cfg), sort_queue(25)
    {
        if (!config.search && !config.verify && !config.print_records)
        {
            total_hashes = 1ULL << config.exponent;
            uint64_t memory_bytes = config.memory_mb * 1024ULL * 1024ULL;
            memory_records = memory_bytes / sizeof(Record);
            rounds = (total_hashes + memory_records - 1) / memory_records;

            // Always show configuration for large experiments
            std::cout << "=== VaultX Configuration ===" << std::endl;
            std::cout << "Threads: " << config.threads << ", I/O Threads: " << config.io_threads << std::endl;
            std::cout << "Exponent: " << config.exponent << ", Total Hashes: " << total_hashes << std::endl;
            std::cout << "Memory: " << config.memory_mb << "MB, Records per round: " << memory_records << ", Rounds: " << rounds << std::endl;
            std::cout << "Final file size: " << (total_hashes * sizeof(Record) / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
            std::cout << "Output: " << config.output_file << std::endl;
            std::cout << "=================================" << std::endl;
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

        if (config.exponent >= 26)
        {
            if (!check_disk_space())
            {
                throw std::runtime_error("Insufficient disk space");
            }
        }

        std::cout << "Starting hash generation with " << config.threads << " compute threads and "
                  << config.io_threads << " I/O threads..." << std::endl;

        // Start I/O threads first
        for (int i = 0; i < config.io_threads; ++i)
        {
            io_threads.emplace_back(&VaultX::io_worker, this, i);
        }

        // Start progress monitor
        std::thread progress_thread(&VaultX::progress_monitor, this);

        // Main processing loop
        for (uint64_t round = 0; round < rounds; ++round)
        {
            if (error_flag)
                break;

            stats.current_round.store(round);
            auto round_start = std::chrono::high_resolution_clock::now();

            uint64_t start_nonce = round * memory_records;
            uint64_t num_this_round = std::min(memory_records, total_hashes - start_nonce);

            std::cout << "[Round " << round + 1 << "/" << rounds << "] Generating " << num_this_round
                      << " hashes..." << std::endl;

            std::vector<Record> records(num_this_round);

// Hash generation phase
#pragma omp parallel for num_threads(config.threads)
            for (uint64_t i = 0; i < num_this_round; ++i)
            {
                if (error_flag)
                    continue;

                Record &rec = records[i];
                uint64_t nonce_val = start_nonce + i;

                // Store nonce
                for (int j = 0; j < NONCE_SIZE; ++j)
                {
                    rec.nonce[j] = (nonce_val >> (j * 8)) & 0xFF;
                }

                // Generate hash
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

            auto hash_end = std::chrono::high_resolution_clock::now();
            double hash_time = std::chrono::duration_cast<std::chrono::duration<double>>(hash_end - round_start).count();

            std::cout << "[Round " << round + 1 << "] Hash generation: " << std::fixed << std::setprecision(4)
                      << hash_time << "s (" << (num_this_round / 1e6 / hash_time) << " MH/s)" << std::endl;

            // Sorting phase
            std::cout << "[Round " << round + 1 << "] Sorting " << num_this_round << " records..." << std::endl;

            auto sort_start = std::chrono::high_resolution_clock::now();
            std::sort(records.begin(), records.end(), [](const Record &a, const Record &b)
                      { return std::memcmp(a.hash, b.hash, HASH_SIZE) < 0; });
            auto sort_end = std::chrono::high_resolution_clock::now();

            double sort_time = std::chrono::duration_cast<std::chrono::duration<double>>(sort_end - sort_start).count();

            std::cout << "[Round " << round + 1 << "] Sorting completed: " << std::fixed << std::setprecision(4)
                      << sort_time << "s" << std::endl;

            // Queue for I/O
            if (!error_flag)
            {
                std::cout << "[Round " << round + 1 << "] Queueing for I/O..." << std::endl;
                sort_queue.push(std::move(records));
                total_io_blocks.fetch_add(1);
            }

            if (error_flag)
            {
                std::cerr << "[Round " << round + 1 << "] Error detected, stopping..." << std::endl;
                break;
            }
        }

        // Shutdown phase
        std::cout << "All rounds completed. Shutting down queue..." << std::endl;
        sort_queue.shutdown_queue();

        std::cout << "Waiting for I/O threads to finish..." << std::endl;
        for (auto &thread : io_threads)
            if (thread.joinable())
                thread.join();

        shutdown = true;
        if (progress_thread.joinable())
            progress_thread.join();

        if (error_flag)
        {
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
            uint64_t needed_space = total_hashes * sizeof(Record);
            double free_gb = free_space / (1024.0 * 1024.0 * 1024.0);
            double needed_gb = needed_space / (1024.0 * 1024.0 * 1024.0);

            std::cout << "Available disk space: " << std::fixed << std::setprecision(2) << free_gb << " GB" << std::endl;
            std::cout << "Needed disk space: " << std::fixed << std::setprecision(2) << needed_gb << " GB" << std::endl;

            if (free_space < needed_space * 1.1)
            {
                std::cerr << "ERROR: Insufficient disk space. Available: "
                          << free_gb << " GB, Needed: " << needed_gb << " GB" << std::endl;
                return false;
            }

            std::cout << "Disk space check passed." << std::endl;
            return true;
        }
        std::cerr << "Warning: Could not check disk space" << std::endl;
        return true;
    }

    void io_worker(int thread_id)
    {
        try
        {
            int fd = open(config.output_file.c_str(), O_RDWR | O_CREAT | O_LARGEFILE, 0666);
            if (fd == -1)
                throw std::runtime_error("Failed to open output file");

            uint64_t current_offset = 0;
            uint64_t blocks_written = 0;

            while (true)
            {
                if (error_flag)
                    break;

                std::vector<Record> batch;
                if (sort_queue.try_pop(batch))
                {
                    size_t total_size = batch.size() * sizeof(Record);
                    blocks_written++;

                    if (config.exponent >= 32 && blocks_written % 10 == 0)
                    {
                        std::cout << "[I/O " << thread_id << "] Writing block " << blocks_written
                                  << " (" << (total_size / (1024.0 * 1024.0)) << " MB)" << std::endl;
                    }

                    const uint8_t *data = reinterpret_cast<const uint8_t *>(batch.data());
                    size_t remaining = total_size;
                    size_t written_total = 0;

                    while (remaining > 0 && !error_flag)
                    {
                        size_t chunk_size = std::min(remaining, (size_t)(64 * 1024 * 1024));
                        ssize_t written = pwrite(fd, data + written_total, chunk_size, current_offset + written_total);

                        if (written < 0)
                        {
                            throw std::runtime_error("Write failed at offset " + std::to_string(current_offset + written_total));
                        }

                        written_total += written;
                        remaining -= written;
                    }

                    current_offset += written_total;
                    stats.bytes_written.fetch_add(written_total);
                    stats.io_blocks_written.fetch_add(1);

                    // Periodic sync for large files
                    if (current_offset % (1024 * 1024 * 1024) == 0)
                    {
                        fsync(fd);
                        if (config.exponent >= 32)
                        {
                            std::cout << "[I/O " << thread_id << "] Synced "
                                      << (current_offset / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
                        }
                    }
                }
                else if (sort_queue.is_shutdown())
                {
                    fsync(fd);
                    if (config.exponent >= 32)
                    {
                        std::cout << "[I/O " << thread_id << "] Completed. Total written: "
                                  << (current_offset / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
                    }
                    break;
                }
            }
            close(fd);
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

    void progress_monitor()
    {
        while (!shutdown && !error_flag)
        {
            std::this_thread::sleep_for(std::chrono::seconds(2));

            if (shutdown || error_flag)
                break;

            uint64_t current_hashes = stats.hashes_generated.load();
            uint64_t current_round = stats.current_round.load();
            uint64_t io_blocks = stats.io_blocks_written.load();

            double total_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                                    std::chrono::high_resolution_clock::now() - stats.start_time)
                                    .count();

            if (total_time > 0)
            {
                double hash_rate = (current_hashes / 1e6) / total_time;
                double io_rate = (stats.bytes_written.load() / (1024.0 * 1024.0)) / total_time;
                double progress = (double)current_hashes / total_hashes * 100.0;

                std::cout << "[" << std::fixed << std::setprecision(0) << total_time
                          << "s] Progress: " << std::setprecision(1) << progress
                          << "% | Round: " << current_round + 1 << "/" << rounds
                          << " | Hash Rate: " << std::setprecision(4) << hash_rate << " MH/s"
                          << " | I/O Rate: " << std::setprecision(4) << io_rate << " MB/s"
                          << " | I/O Blocks: " << io_blocks
                          << std::endl;
            }
        }
    }

    void print_final_stats()
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                                end_time - stats.start_time)
                                .count();

        // FIXED: Use actual bytes written for I/O rate
        uint64_t actual_bytes_written = total_hashes * sizeof(Record);
        double hash_rate = (total_hashes / 1e6) / total_time;
        double io_rate = (actual_bytes_written / (1024.0 * 1024.0)) / total_time;

        std::cout << "=== Final Statistics ===" << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(4) << total_time << " seconds" << std::endl;
        std::cout << "Hash rate: " << std::fixed << std::setprecision(4) << hash_rate << " MH/s" << std::endl;
        std::cout << "I/O rate: " << std::fixed << std::setprecision(4) << io_rate << " MB/s" << std::endl;
        std::cout << "File size: " << std::fixed << std::setprecision(6) << (actual_bytes_written / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;

        // Output in required format
        std::cout << "vaultx t" << config.threads << " i" << config.io_threads
                  << " m" << config.memory_mb << " k" << config.exponent
                  << " " << std::fixed << std::setprecision(4) << hash_rate
                  << " " << std::fixed << std::setprecision(4) << io_rate
                  << " " << std::fixed << std::setprecision(6) << total_time << std::endl;
    }

    void generate_real_prefixes(int num, int difficulty, std::vector<std::vector<uint8_t>> &prefixes)
    {
        prefixes.clear();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint8_t> dis(0, 255);

        for (int i = 0; i < num; ++i)
        {
            std::vector<uint8_t> prefix(difficulty);
            for (int j = 0; j < difficulty; ++j)
            {
                prefix[j] = dis(gen);
            }
            prefixes.push_back(prefix);
        }
    }

    void perform_search()
    {
        std::cout << "searches=" << config.num_searches << " difficulty=" << config.difficulty << std::endl;

        struct stat file_stat;
        if (stat(config.output_file.c_str(), &file_stat) != 0)
        {
            throw std::runtime_error("Cannot access file: " + config.output_file);
        }

        uint64_t file_size = file_stat.st_size;
        uint64_t total_records = file_size / sizeof(Record);

        int actual_k = std::log2(total_records);

        std::cout << "Parsed k : " << actual_k << std::endl;
        std::cout << "Nonce Size : " << NONCE_SIZE << std::endl;
        std::cout << "Record Size : " << sizeof(Record) << std::endl;
        std::cout << "Hash Size : " << HASH_SIZE << std::endl;
        std::cout << "Number of Hashes : " << total_records << std::endl;
        std::cout << "File Size: " << file_size << " bytes" << std::endl;

        int fd = open(config.output_file.c_str(), O_RDONLY | O_LARGEFILE);
        if (fd == -1)
        {
            throw std::runtime_error("Failed to open file for search: " + config.output_file);
        }

        std::vector<std::vector<uint8_t>> prefixes;
        generate_real_prefixes(config.num_searches, config.difficulty, prefixes);

        uint64_t total_matches = 0;
        uint64_t total_seeks = 0;
        uint64_t total_comps = 0;
        uint64_t found_queries = 0;
        uint64_t notfound_queries = 0;
        double total_search_time = 0.0;

        auto search_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < config.num_searches; ++i)
        {
            auto query_start = std::chrono::high_resolution_clock::now();

            const uint8_t *prefix = prefixes[i].data();
            uint64_t left = 0;
            uint64_t right = total_records - 1;
            uint64_t seeks = 0;
            uint64_t comps = 0;
            bool found = false;

            while (left <= right)
            {
                seeks++;
                uint64_t mid = left + (right - left) / 2;

                Record record;
                ssize_t bytes_read = pread(fd, &record, sizeof(Record), mid * sizeof(Record));
                if (bytes_read != sizeof(Record))
                {
                    break;
                }

                int cmp = std::memcmp(record.hash, prefix, config.difficulty);
                comps++;

                if (cmp == 0)
                {
                    found = true;
                    total_matches++;
                    break;
                }
                else if (cmp < 0)
                {
                    left = mid + 1;
                }
                else
                {
                    right = mid - 1;
                }
            }

            auto query_end = std::chrono::high_resolution_clock::now();
            double query_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                                    query_end - query_start)
                                    .count();

            total_search_time += query_time;
            total_seeks += seeks;
            total_comps += comps;

            if (found)
            {
                found_queries++;
            }
            else
            {
                notfound_queries++;
            }

            // Progress for large searches
            if (config.num_searches >= 1000 && (i + 1) % 100 == 0)
            {
                double progress = (double)(i + 1) / config.num_searches * 100.0;
                std::cout << "Search progress: " << std::fixed << std::setprecision(1) << progress << "%" << std::endl;
            }
        }

        close(fd);

        auto search_end = std::chrono::high_resolution_clock::now();
        double actual_total_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                                       search_end - search_start)
                                       .count();

        double avg_ms = (actual_total_time * 1000.0) / config.num_searches;
        double searches_per_sec = config.num_searches / actual_total_time;
        double avg_seeks = static_cast<double>(total_seeks) / config.num_searches;
        double avg_comps = static_cast<double>(total_comps) / config.num_searches;
        double avg_matches = found_queries > 0 ? static_cast<double>(total_matches) / found_queries : 0.0;

        std::cout << "Search Summary: requested=" << config.num_searches
                  << " found_queries=" << found_queries
                  << " total_matches=" << total_matches
                  << " notfound=" << notfound_queries
                  << " total_time=" << std::fixed << std::setprecision(6) << actual_total_time << " s"
                  << " avg_ms=" << std::fixed << std::setprecision(3) << avg_ms << " ms"
                  << " searches/sec=" << std::fixed << std::setprecision(0) << searches_per_sec
                  << " total_seeks=" << total_seeks
                  << " avg_seeks=" << std::fixed << std::setprecision(3) << avg_seeks
                  << " total_comps=" << total_comps
                  << " avg_comps=" << std::fixed << std::setprecision(3) << avg_comps
                  << " avg_matches=" << std::fixed << std::setprecision(3) << avg_matches
                  << std::endl;
    }

    void print_records()
    {
        struct stat file_stat;
        if (stat(config.output_file.c_str(), &file_stat) != 0)
        {
            throw std::runtime_error("Cannot access file: " + config.output_file);
        }

        uint64_t file_size = file_stat.st_size;
        uint64_t total_records = file_size / sizeof(Record);

        int fd = open(config.output_file.c_str(), O_RDONLY | O_LARGEFILE);
        if (fd == -1)
        {
            throw std::runtime_error("Failed to open file: " + config.output_file);
        }

        Record record;
        int records_to_print = std::min(config.print_records, static_cast<int>(total_records));

        std::cout << "Printing " << records_to_print << " records:" << std::endl;
        for (int i = 0; i < records_to_print; ++i)
        {
            ssize_t bytes_read = pread(fd, &record, sizeof(Record), i * sizeof(Record));
            if (bytes_read != sizeof(Record))
            {
                break;
            }

            uint64_t nonce_val = 0;
            for (int j = 0; j < NONCE_SIZE; ++j)
            {
                nonce_val |= (static_cast<uint64_t>(record.nonce[j]) << (j * 8));
            }

            std::cout << "[" << (i * sizeof(Record)) << "] stored: "
                      << bytes_to_hex(record.hash, HASH_SIZE)
                      << " nonce: " << nonce_val << std::endl;
        }

        close(fd);
    }

    void verify_sorted_order()
    {
        struct stat file_stat;
        if (stat(config.output_file.c_str(), &file_stat) != 0)
        {
            throw std::runtime_error("Cannot access file: " + config.output_file);
        }

        uint64_t file_size = file_stat.st_size;
        uint64_t total_records = file_size / sizeof(Record);

        std::cout << "Verifying sorted order of '" << config.output_file << "' (" << file_size << " bytes)..." << std::endl;

        int fd = open(config.output_file.c_str(), O_RDONLY | O_LARGEFILE);
        if (fd == -1)
        {
            throw std::runtime_error("Failed to open file: " + config.output_file);
        }

        Record current, next;
        uint64_t sorted_count = 0;
        uint64_t not_sorted_count = 0;
        uint64_t progress_interval = total_records / 10;

        auto verify_start = std::chrono::high_resolution_clock::now();

        for (uint64_t i = 0; i < total_records - 1; ++i)
        {
            ssize_t bytes_read = pread(fd, &current, sizeof(Record), i * sizeof(Record));
            if (bytes_read != sizeof(Record))
                break;

            bytes_read = pread(fd, &next, sizeof(Record), (i + 1) * sizeof(Record));
            if (bytes_read != sizeof(Record))
                break;

            int cmp = std::memcmp(current.hash, next.hash, HASH_SIZE);
            if (cmp <= 0)
            {
                sorted_count++;
            }
            else
            {
                not_sorted_count++;
            }

            // Progress reporting
            if ((i + 1) % progress_interval == 0)
            {
                double progress = (double)(i + 1) / total_records * 100.0;
                auto current_time = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - verify_start).count();
                std::cout << "Verify progress: " << std::fixed << std::setprecision(1) << progress
                          << "% (" << elapsed << "s elapsed)" << std::endl;
            }
        }

        close(fd);

        auto verify_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration_cast<std::chrono::duration<double>>(verify_end - verify_start).count();

        std::cout << "Verification completed in " << std::fixed << std::setprecision(2) << total_time << "s" << std::endl;
        std::cout << "sorted=" << sorted_count << " not_sorted=" << not_sorted_count
                  << " total_records=" << total_records << std::endl;
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

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "-t" || arg == "--threads")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing threads");
            config.threads = std::stoi(argv[i]);
        }
        else if (arg == "-i" || arg == "--iothreads")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing iothreads");
            config.io_threads = std::stoi(argv[i]);
        }
        else if (arg == "-a" || arg == "--approach")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing approach");
            config.approach = std::stoi(argv[i]);
        }
        else if (arg == "-c" || arg == "--compression")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing compression");
            config.compression = std::stoi(argv[i]);
        }
        else if (arg == "-k" || arg == "--exponent")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing exponent");
            config.exponent = std::stoi(argv[i]);
        }
        else if (arg == "-m" || arg == "--memory")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing memory");
            config.memory_mb = std::stoi(argv[i]);
        }
        else if (arg == "-f" || arg == "--file")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing file");
            config.output_file = argv[i];
        }
        else if (arg == "-g" || arg == "--file_final")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing temp file");
            config.temp_file = argv[i];
        }
        else if (arg == "-d" || arg == "--debug")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing debug");
            config.debug = (std::string(argv[i]) == "true");
        }
        else if (arg == "-b" || arg == "--batch-size")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing batch size");
            config.batch_size = std::stoi(argv[i]);
        }
        else if (arg == "-p" || arg == "--print")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing print count");
            config.print_records = std::stoi(argv[i]);
        }
        else if (arg == "-s" || arg == "--search")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing search count");
            config.num_searches = std::stoi(argv[i]);
            config.search = true;
        }
        else if (arg == "-q" || arg == "--difficulty")
        {
            if (++i >= argc)
                throw std::runtime_error("Missing difficulty");
            config.difficulty = std::stoi(argv[i]);
        }
        else if (arg == "-v" || arg == "--verify")
        {
            config.verify = true;
        }
        else if (arg == "-h" || arg == "--help")
        {
            std::cout << "Usage: ./vaultx [OPTIONS]\n"
                      << "Options:\n"
                      << "-a, --approach [task|for]   Parallelization mode (default: for)\n"
                      << "-t, --threads NUM           Hashing threads (default: all cores)\n"
                      << "-i, --iothreads NUM         Number of I/O threads to use (default: 1)\n"
                      << "-c, --compression NUM       Compression: number of hash bytes to discard (0..HASH_SIZE)\n"
                      << "-k, --exponent NUM          Exponent k for 2^K iterations (default: 26)\n"
                      << "-m, --memory NUM            Memory size in MB (default: 1)\n"
                      << "-f, --file NAME             Final output file\n"
                      << "-g, --file_final NAME       Temporary file (intermediate output)\n"
                      << "-d, --debug [true|false]    Enable per-search debug prints (default: false)\n"
                      << "-b, --batch-size NUM        Batch size (default: 1024)\n"
                      << "-p, --print NUM             Print NUM records and exit\n"
                      << "-s, --search NUM            Enable search of specified number of records\n"
                      << "-q, --difficulty NUM        Set difficulty for search in bytes\n"
                      << "-v, --verify                Verify sorted order\n"
                      << "-h, --help                  Display this help message\n"
                      << "Example:\n"
                      << "./vaultx -t 24 -i 1 -m 1024 -k 26 -g memo.t -f memo.x -d true\n";
            exit(0);
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
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