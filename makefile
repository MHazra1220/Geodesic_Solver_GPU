main.sh: main.o photon.o
	nvcc -o main.sh main.o photon.o

main.o: main.cu
	nvcc -c main.cu -dc

photon.o: photon.h photon.cu
	nvcc -c photon.cu -dc

make clean:
	rm main.o photon.o
