all: MultiGA2 python/sparse_autoencoder.pyc mnist.csv

mnist.csv: python/preprocess.pyc train.csv
	python $<

obj/%.o: cpp/%.cpp 
	g++ -std=c++11 -O3 -march=native -c -o $@ $< 

MultiGA2: obj/MultiGA2.o obj/myrand.o obj/mt19937ar.o
	g++ -std=c++11 -O3 -march=native -o $@ $^

python/%.pyc: python/%.py
	python -m py_compile $^

run: all
	./MultiGA2

clean:
	rm -f obj/*
	rm -f MultiGA2
	rm -f python/*.pyc
	rm -f mnist.csv

.PHONY: clean run all
