$/bin/sh

./MultiGA2 49 40 I49O40 mnist.csv

for j in 1 2 3
do
	./MultiGA2 40 30 I40O30J${j} data/output_I49O40-${j}
done

./shrink I49O40 I40O30


for j in 1 2 3
do
	./MultiGA2 30 20 I30O20J${j} data/output_I40O30-${j}
done

./shrink I40O30 I30O20

