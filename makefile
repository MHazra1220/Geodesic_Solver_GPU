main.sh: main.o scene.o
	nvcc -o main.sh main.o scene.o

main.o: main.cu
	nvcc -c main.cu

scene.o: scene.h scene.cu
	nvcc -c scene.cu -diag-suppress 550

make clean:
	rm main.o scene.o
