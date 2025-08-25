main.sh: main.o world.o particle.o camera.o
	nvcc -o main.sh main.o world.o particle.o camera.o -Xcompiler -fopenmp -O1

main.o: main.cu
	nvcc -c main.cu -O1

world.o: world.h world.cpp
	nvcc -c world.cpp -O1

particle.o: particle.h particle.cpp
	nvcc -c particle.cpp -O1

camera.o: camera.h camera.cpp
	nvcc -c camera.cpp -Xcompiler -fopenmp -O1

clean:
	rm main.o world.o particle.o camera.o
