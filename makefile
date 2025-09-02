main.sh: main.o scene.o
	nvcc -o main.sh main.o scene.o -O1 -Xptxas -O1

main.o: main.cu
	nvcc -c main.cu -O1 -Xptxas -O1

scene.o: scene.h scene.cu
	nvcc -c scene.cu -diag-suppress 550 -O1 -Xptxas -O1

make clean:
	rm main.o scene.o
