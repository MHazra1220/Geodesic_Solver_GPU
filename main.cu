#include "photon.h"
#include <iostream>

__global__ void normalisePhotonVelocities(Photon photons[], int width, int height)
{
    int photon_num = blockDim.x * blockIdx.x + threadIdx.x;
    if (photon_num < width*height)
    {
        photons[photon_num].v[1] = 1.;
        photons[photon_num].v[2] = 1.;
        photons[photon_num].v[3] = 1.;
        photons[photon_num].setMetric();
        photons[photon_num].makeVNull();
    }
}

int main()
{
    // CUDA photon allocation test.
    cudaError_t err { cudaSuccess };

    int width { 2560 };
    int height { 1440 };
    const int num_photons { width*height };

    // Allocate memory on host.
    size_t photon_array_size { num_photons * sizeof(Photon) };
    Photon *host_photons { (Photon*)malloc(photon_array_size) };

    // Allocate device memory.
    Photon *device_photons { nullptr };
    err = cudaMalloc((void **)&device_photons, photon_array_size);
    err = cudaMemcpy(device_photons, host_photons, photon_array_size, cudaMemcpyHostToDevice);

    // Call kernel.
    int threadsPerBlock { 256 };
    int numBlocks { (num_photons + threadsPerBlock - 1) / threadsPerBlock };
    normalisePhotonVelocities<<<numBlocks, threadsPerBlock>>>(device_photons, width, height);
    err = cudaGetLastError();

    // Transfer back to host.
    err = cudaMemcpy(host_photons, device_photons, photon_array_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cout << "FAILED" << "\n";
    }

    std::cout << host_photons[0].v[0] << "\n";
    std::cout << host_photons[0].v[1] << "\n";
    std::cout << host_photons[0].v[2] << "\n";
    std::cout << host_photons[0].v[3] << "\n";

    // Free host memory.
    free(host_photons);

    // Free device memory.
    err = cudaFree(device_photons);

    return 0;
}
