CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -pthread -fopenmp -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE
LDFLAGS = -pthread -lcrypto -L/usr/local/lib -lblake3 -lstdc++ -fopenmp

TARGET = vaultx
SOURCES = vaultx.cpp

.PHONY: all clean run-small run-large run-search run-all test-small run-large-single

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.t *.x *.csv *.png results_*.txt test_results.txt

run-small:
	@echo "=== Running K=26 experiments ===" | tee results_small.txt
	@echo "This will run 42 experiments (3 memory x 7 threads x 2 I/O threads)" | tee -a results_small.txt
	@echo "Estimated time: 60-90 minutes" | tee -a results_small.txt
	@echo "Started: $(shell date)" | tee -a results_small.txt
	@echo "" | tee -a results_small.txt
	@for memory in 256 512 1024; do \
		for threads in 1 2 4 8 12 24 48; do \
			for iothreads in 1 2 4; do \
				f=k26-t$$threads-i$$iothreads-m$$memory.x; \
				echo "=== Running: threads=$$threads, memory=$$memory, iothreads=$$iothreads ===" | tee -a results_small.txt; \
				echo "=== threads=$$threads, memory=$$memory, iothreads=$$iothreads ==="; \
				./$(TARGET) -t $$threads -i $$iothreads -m $$memory -k 26 -g temp.t -f $$f 2>&1 | tee -a results_small.txt; \
				echo "Verifying..." | tee -a results_small.txt; \
				./$(TARGET) -f $$f -v 2>&1 | tee -a results_small.txt; \
				echo "---" | tee -a results_small.txt; \
				rm -f temp.t $$f; \
			done; \
		done; \
	done
	@echo "K=26 experiments completed at: $(shell date)" | tee -a results_small.txt

run-large:
	@echo "=== Running K=32 large workload ===" | tee results_large.txt
	@echo "This will run 7 experiments (7 memory sizes)" | tee -a results_large.txt
	@echo "Estimated time: 5-10 hours" | tee -a results_large.txt
	@echo "Started: $(shell date)" | tee -a results_large.txt
	@echo "" | tee -a results_large.txt
	@for memory in 1024 2048 4096 8192 16384 32768 65536; do \
		f=k32-t24-i1-m$$memory.x; \
		echo "=== Running: threads=24, memory=$$memory, iothreads=1 ===" | tee -a results_large.txt; \
		echo "=== threads=24, memory=$$memory, iothreads=1 ==="; \
		./$(TARGET) -t 24 -i 1 -m $$memory -k 32 -g temp.t -f $$f 2>&1 | tee -a results_large.txt; \
		echo "Verifying..." | tee -a results_large.txt; \
		./$(TARGET) -f $$f -v 2>&1 | tee -a results_large.txt; \
		echo "---" | tee -a results_large.txt; \
		rm -f temp.t $$f; \
	done
	@echo "K=32 large workload completed at: $(shell date)" | tee -a results_large.txt

run-large-safe:
	@echo "=== Running K=32 large workload (SAFE MODE - reduced memory range) ===" | tee results_large.txt
	@echo "This will run 3 experiments (3 memory sizes)" | tee -a results_large.txt
	@echo "Estimated time: 3-6 hours" | tee -a results_large.txt
	@echo "Started: $(shell date)" | tee -a results_large.txt
	@echo "" | tee -a results_large.txt
	@for memory in 1024 2048 4096; do \
		f=k32-t24-i1-m$$memory.x; \
		echo "=== Running: threads=24, memory=$$memory, iothreads=1 ===" | tee -a results_large.txt; \
		echo "=== threads=24, memory=$$memory, iothreads=1 ==="; \
		./$(TARGET) -t 24 -i 1 -m $$memory -k 32 -g temp.t -f $$f 2>&1 | tee -a results_large.txt; \
		echo "Verifying..." | tee -a results_large.txt; \
		./$(TARGET) -f $$f -v 2>&1 | tee -a results_large.txt; \
		echo "---" | tee -a results_large.txt; \
		rm -f temp.t $$f; \
	done
	@echo "K=32 large workload completed at: $(shell date)" | tee -a results_large.txt

run-search:
	@echo "=== Running search experiments ===" | tee results_search.txt
	@echo "Started: $(shell date)" | tee -a results_search.txt
	@echo "" | tee -a results_search.txt
	@echo "Generating files for search..." | tee -a results_search.txt
	@./$(TARGET) -t 24 -i 1 -m 1024 -k 26 -g temp.t -f k26-t24-i1-m1024.x 2>&1 | tee -a results_search.txt
	@./$(TARGET) -t 24 -i 1 -m 2048 -k 32 -g temp.t -f k32-t24-i1-m2048.x 2>&1 | tee -a results_search.txt
	@rm -f temp.t
	@echo "Verifying search files..." | tee -a results_search.txt
	@./$(TARGET) -f k26-t24-i1-m1024.x -v 2>&1 | tee -a results_search.txt
	@./$(TARGET) -f k32-t24-i1-m2048.x -v 2>&1 | tee -a results_search.txt
	@echo "Running searches..." | tee -a results_search.txt
	@for k in 26 32; do \
		for q in 3 4 5; do \
			echo "=== k=$$k, difficulty=$$q ===" | tee -a results_search.txt; \
			echo "=== k=$$k, difficulty=$$q ==="; \
			if [ "$$k" = "26" ]; then \
				f=k26-t24-i1-m1024.x; \
			else \
				f=k32-t24-i1-m2048.x; \
			fi; \
			./$(TARGET) -k $$k -f $$f -s 1000 -q $$q 2>&1 | tee -a results_search.txt; \
			echo "---" | tee -a results_search.txt; \
		done; \
	done
	@rm -f k26-t24-i1-m1024.x k32-t24-i1-m2048.x
	@echo "Search experiments completed at: $(shell date)" | tee -a results_search.txt

run-all: run-small run-large-safe run-search
	@echo "All experiments completed at: $(shell date)"

run-large-single:
	@echo "Running single K=32 experiment with memory=2048..."
	@./$(TARGET) -t 24 -i 1 -m 2048 -k 32 -g temp.t -f k32-t24-i1-m2048.x -d false
	@rm -f temp.t k32-t24-i1-m2048.x

test-small:
	@echo "Testing with K=10 (small test)..."
	@./$(TARGET) -t 4 -i 1 -m 16 -k 10 -g test.t -f test.x -d true
	@./$(TARGET) -f test.x -v
	@./$(TARGET) -f test.x -p 10
	@rm -f test.t test.x

run-small-quick:
	@echo "=== Running quick K=26 tests (subset) ===" | tee results_small_quick.txt
	@echo "Started: $(shell date)" | tee -a results_small_quick.txt
	@echo "" | tee -a results_small_quick.txt
	@for memory in 256 1024; do \
		for threads in 1 4 24; do \
			for iothreads in 1 4; do \
				f=k26-t$$threads-i$$iothreads-m$$memory.x; \
				echo "=== Running: threads=$$threads, memory=$$memory, iothreads=$$iothreads ===" | tee -a results_small_quick.txt; \
				echo "=== threads=$$threads, memory=$$memory, iothreads=$$iothreads ==="; \
				./$(TARGET) -t $$threads -i $$iothreads -m $$memory -k 26 -g temp.t -f $$f 2>&1 | tee -a results_small_quick.txt; \
				echo "Verifying..." | tee -a results_small_quick.txt; \
				./$(TARGET) -f $$f -v 2>&1 | tee -a results_small_quick.txt; \
				echo "---" | tee -a results_small_quick.txt; \
				rm -f temp.t $$f; \
			done; \
		done; \
	done
	@echo "Quick K=26 tests completed at: $(shell date)" | tee -a results_small_quick.txt

status:
	@echo "=== Current Status ==="
	@if [ -f results_small.txt ]; then \
		echo "Small experiments: $(shell grep -c "threads=" results_small.txt || echo 0)/42 completed"; \
	else \
		echo "Small experiments: Not started"; \
	fi
	@if [ -f results_large.txt ]; then \
		echo "Large experiments: $(shell grep -c "threads=24, memory=" results_large.txt || echo 0)/7 completed"; \
	else \
		echo "Large experiments: Not started"; \
	fi
	@if [ -f results_search.txt ]; then \
		echo "Search experiments: $(shell grep -c "k=" results_search.txt || echo 0)/6 completed"; \
	else \
		echo "Search experiments: Not started"; \
	fi
	@echo "Disk space: $(shell df -h . | awk 'NR==2 {print $4}') available"