main.sh: main.o scene.o quaternion.o
	nvcc -o main.sh main.o scene.o quaternion.o

main.o: main.cu
	nvcc -c main.cu

scene.o: scene.h scene.cu
	nvcc -c scene.cu -diag-suppress 550

quaternion.o: quaternion.h quaternion.cu
	nvcc -c quaternion.cu

make clean:
	rm main.o scene.o quaternion.o
