#include "scene.h"
#include <algorithm>
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

    float metric_func[4][4];
    float metric_inverse[4][4];

    // Define some test 4x4 symmetric matrix.
    metric_func[0][0] = 2.;
    metric_func[1][1] = 1.2;
    metric_func[2][2] = -2.;
    metric_func[3][3] = 0.5;
    metric_func[0][1] = -0.4;
    metric_func[1][0] = -2.;
    metric_func[0][2] = 0.;
    metric_func[2][0] = 0.;
    metric_func[0][3] = -2.5;
    metric_func[3][0] = -2.5;
    metric_func[1][2] = -1.;
    metric_func[2][1] = -1.;
    metric_func[1][3] = 0.;
    metric_func[3][1] = 0.;
    metric_func[2][3] = 1.;
    metric_func[3][2] = 1.;

    for (int i { 0 }; i < 4; i++)
    {
        for (int j { 0 }; j < 4; j++)
        {
            std::cout << metric_func[i][j] << "\t";
            // Set metric_inverse to the identity.
            if (i == j)
            {
                metric_inverse[i][j] = 1.;
            }
            else
            {
                metric_inverse[i][j] = 0.;
            }

        }
        std::cout << "\n";
    }

    invertMetric(metric_func, metric_inverse);

    for (int i { 0 }; i < 4; i++)
    {
        for (int j { 0 }; j < 4; j++)
        {
            std::cout << metric_inverse[i][j] << "\t";

        }
        std::cout << "\n";
    }

    return 0;
}
