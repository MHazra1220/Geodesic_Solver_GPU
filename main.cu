#include "scene.h"
#include <cmath>
#include <iostream>

int main()
{
    // // CUDA photon allocation test.
    // cudaError_t err { cudaSuccess };
    //
    // // Allocate memory on host.
    // size_t photon_array_size { num_photons * sizeof(Photon) };
    // Photon *host_photons { (Photon*)malloc(photon_array_size) };
    //
    // // Allocate device memory.
    // Photon *device_photons { nullptr };
    // err = cudaMalloc((void **)&device_photons, photon_array_size);
    // err = cudaMemcpy(device_photons, host_photons, photon_array_size, cudaMemcpyHostToDevice);
    //
    // // Call kernel.
    // int threadsPerBlock { 128 };
    // int numBlocks { (num_photons + threadsPerBlock - 1) / threadsPerBlock };
    // normalisePhotonVelocities<<<numBlocks, threadsPerBlock>>>(device_photons, width, height);
    //
    // // Transfer back to host.
    // err = cudaMemcpy(host_photons, device_photons, photon_array_size, cudaMemcpyDeviceToHost);
    //
    // std::cout << host_photons[1204].v[0] << "\n";
    // std::cout << host_photons[1204].v[1] << "\n";
    // std::cout << host_photons[1204].v[2] << "\n";
    // std::cout << host_photons[1204].v[3] << "\n";
    //
    // // Free host memory.
    // free(host_photons);
    //
    // // Free device memory.
    // err = cudaFree(device_photons);

    char sky_map[] { "/media/mh2001/SSD2/Programming/General Relativity/Geodesic_Solver_GPU/sky_box_samples/full_milky_way.jpg" };
    Scene scene_test;
    scene_test.initialiseDefault(sky_map);
    float pos[4] { 0., 20, 0., 0. };
    float quat[4] { 0., 0., 0., 1. };
    scene_test.setCameraCoordinates(pos);
    scene_test.setCameraQuaternion(quat);
    scene_test.setCameraRes(2560, 1440);
    scene_test.setCameraFoV(90.);
    scene_test.runTraceKernel();
    char output_image[] { "/media/mh2001/SSD2/Programming/General Relativity/Geodesic_Solver_GPU/output_images/GPU_test.jpg" };
    scene_test.writeCameraImage(output_image);
    scene_test.freeHostPixelArrays();
    scene_test.freeDevicePixelArrays();

    return 0;
}
