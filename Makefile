all: MultiGA2 shrink python/sparse_autoencoder.pyc

mnist.csv: python/preprocess.pyc train.csv
	python $<

mnist_label.csv: python/preprocess.pyc train.csv
	python $<

obj/%.o: cpp/%.cpp 
	g++ -std=c++11 -O3 -march=native -c -o $@ $< -fopenmp -lpthread

MultiGA2: obj/MultiGA2.o obj/myrand.o obj/mt19937ar.o
	g++ -std=c++11 -O3 -march=native -o $@ $^ -fopenmp -lpthread

shrink: obj/shrink.o
	g++ -std=c++11 -O3 -march=native -o $@ $^
	#g++ -std=c++11 -g -O0 -march=native -o $@ $^

python/%.pyc: python/%.py
	python -m py_compile $^

run: all
	./run.sh

run_final: python/sparse_nn.pyc mnist.csv mnist_label.csv
	python $<

clean:
	rm -f obj/*
	rm -f MultiGA2
	rm -f python/*.pyc
	rm -f request*
	rm -f reply*
	rm -f spec

.PHONY: clean run
