#include "scene.h"
#include <cmath>
#include <iostream>

int main()
{
    char sky_map[] { "/media/mh2001/SSD2/Programming/General_Relativity/Geodesic_Solver_GPU/sky_box_samples/full_milky_way.jpg" };
    Scene scene_test;
    scene_test.initialiseDefault(sky_map);
    float pos[4] { 0., 20, 0., 0. };
    float quat[4] { 0., 0., 0., 1. };
    scene_test.setCameraCoordinates(pos);
    scene_test.setCameraQuaternion(quat);
    scene_test.setCameraRes(1920, 1080);
    scene_test.setCameraFoV(90.);
    scene_test.runTraceKernel();
    char output_image[] { "/media/mh2001/SSD2/Programming/General_Relativity/Geodesic_Solver_GPU/output_images/GPU_test.jpg" };
    scene_test.writeCameraImage(output_image);
    scene_test.freeHostPixelArrays();
    scene_test.freeDevicePixelArrays();

    return 0;
}
