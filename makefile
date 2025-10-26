CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -pthread -fopenmp -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE
LDFLAGS = -pthread -lcrypto -L/usr/local/lib -lblake3 -lstdc++ -fopenmp

TARGET = vaultx
SOURCES = vaultx.cpp

.PHONY: all clean run-small run-large run-search run-all test-small

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.t *.x *.csv *.png results_*.txt test_results.txt plots/*.png plots/*.csv

run-small:
	@echo "=== Running K=26 experiments ===" | tee results_small.txt
	@echo "This will run 42 experiments (3 memory x 7 threads x 2 I/O threads)" | tee -a results_small.txt
	@echo "Started: $(shell date)" | tee -a results_small.txt
	@echo "" | tee -a results_small.txt
	@for memory in 256 512 1024; do \
		for threads in 1 2 4 8 12 24 48; do \
			for iothreads in 1 2 4; do \
				f=k26-t$$threads-i$$iothreads-m$$memory.x; \
				echo "=== Running: threads=$$threads, memory=$$memory, iothreads=$$iothreads ===" | tee -a results_small.txt; \
				./$(TARGET) -t $$threads -i $$iothreads -m $$memory -k 26 -g temp.t -f $$f -d false 2>&1 | tee -a results_small.txt; \
				echo "---" | tee -a results_small.txt; \
				rm -f temp.t $$f; \
			done; \
		done; \
	done
	@echo "K=26 experiments completed at: $(shell date)" | tee -a results_small.txt

run-large:
	@echo "=== Running K=32 large workload ===" | tee results_large.txt
	@echo "This will run 7 experiments (7 memory sizes)" | tee -a results_large.txt
	@echo "Started: $(shell date)" | tee -a results_large.txt
	@echo "" | tee -a results_large.txt
	@for memory in 1024 2048 4096 8192 16384 32768 65536; do \
		f=k32-t24-i1-m$$memory.x; \
		echo "=== Running: threads=24, memory=$$memory, iothreads=1 ===" | tee -a results_large.txt; \
		./$(TARGET) -t 24 -i 1 -m $$memory -k 32 -g temp.t -f $$f -d false 2>&1 | tee -a results_large.txt; \
		echo "---" | tee -a results_large.txt; \
		rm -f temp.t $$f; \
	done
	@echo "K=32 large workload completed at: $(shell date)" | tee -a results_large.txt

run-search:
	@echo "=== Running search experiments ===" | tee results_search.txt
	@echo "Started: $(shell date)" | tee -a results_search.txt
	@echo "" | tee -a results_search.txt
	@echo "Generating files for search..." | tee -a results_search.txt
	@./$(TARGET) -t 24 -i 1 -m 1024 -k 26 -g temp.t -f k26-search.x -d false 2>&1 | tee -a results_search.txt
	@./$(TARGET) -t 24 -i 1 -m 2048 -k 32 -g temp.t -f k32-search.x -d false 2>&1 | tee -a results_search.txt
	@rm -f temp.t
	@echo "Running searches..." | tee -a results_search.txt
	@for k in 26 32; do \
		for q in 3 4 5; do \
			echo "=== k=$$k, difficulty=$$q ===" | tee -a results_search.txt; \
			if [ "$$k" = "26" ]; then \
				f=k26-search.x; \
			else \
				f=k32-search.x; \
			fi; \
			./$(TARGET) -k $$k -f $$f -s 1000 -q $$q -d false 2>&1 | tee -a results_search.txt; \
			echo "---" | tee -a results_search.txt; \
		done; \
	done
	@rm -f k26-search.x k32-search.x
	@echo "Search experiments completed at: $(shell date)" | tee -a results_search.txt

run-all: run-small run-large run-search
	@echo "All experiments completed at: $(shell date)"

test-small:
	@echo "Testing with K=10 (small test)..."
	@./$(TARGET) -t 4 -i 1 -m 16 -k 10 -g test.t -f test.x -d true
	@./$(TARGET) -f test.x -v
	@./$(TARGET) -f test.x -p 10
	@rm -f test.t test.x

test-verify:
	@echo "Testing verification..."
	@./$(TARGET) -t 4 -i 1 -m 16 -k 10 -g test.t -f test.x -d false
	@./$(TARGET) -f test.x -v
	@rm -f test.t test.x

test-search:
	@echo "Testing search..."
	@./$(TARGET) -t 4 -i 1 -m 16 -k 10 -g test.t -f test.x -d false
	@./$(TARGET) -f test.x -s 10 -q 2 -d true
	@rm -f test.t test.x

plots:
	@python3 results.py

status:
	@echo "=== Current Status ==="
	@if [ -f results_small.txt ]; then \
		echo "Small experiments: $(shell grep -c "vaultx t" results_small.txt || echo 0)/42 completed"; \
	else \
		echo "Small experiments: Not started"; \
	fi
	@if [ -f results_large.txt ]; then \
		echo "Large experiments: $(shell grep -c "vaultx t24" results_large.txt || echo 0)/7 completed"; \
	else \
		echo "Large experiments: Not started"; \
	fi
	@if [ -f results_search.txt ]; then \
		echo "Search experiments: $(shell grep -c "k=" results_search.txt || echo 0)/6 completed"; \
	else \
		echo "Search experiments: Not started"; \
	fi
	@echo "Disk space: $(shell df -h . | awk 'NR==2 {print $4}') available"