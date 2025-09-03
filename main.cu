#include "scene.h"
#include <cmath>
#include <iostream>

void invertMetric(double metric_func[4][4], double metric_inverse[4][4])
{
    // Assume that that there are no zeros on the diagonal of metric_func and that metric_inverse is currently the identity matrix.
    // Invert with forward and backward-propagation (i.e. LU-decomposition). metric_func and metric_temp are both overwritten to avoid memory allocation.
    // WARNING: For now, assume that there are no zeros on the diagonal of the metric (very unlikely in t, x, y, z coordinates).

    double multiplier;

    // Forward-propagation pass.
    for (int i { 0 }; i < 3; i++)
    {
        for (int j { i+1 }; j < 4; j++)
        {
            multiplier = metric_func[j][i] / metric_func[i][i];
            // Use the accumulating zeros in the lower-triangular half to reduce the number of calculations.
            for (int k { i }; k < 4; k++)
            {
                metric_func[j][k] -= multiplier*metric_func[i][k];
            }
            // Use the fact that metric_inverse is currently an identity matrix to reduce the number of calculations.
            for (int k { 0 }; k < j; k++)
            {
                metric_inverse[j][k] -= multiplier*metric_inverse[i][k];
            }
        }
    }

    // Backward-propagation pass.
    for (int i { 3 }; i > 0; i--)
    {
        for (int j { i-1 }; j > -1; j--)
        {
            multiplier = metric_func[j][i] / metric_func[i][i];
            // Use the zeros in the lower-triangular half of metric_func to reduce the number of calculations.
            for (int k { i }; k < 4; k++)
            {
                metric_func[j][k] -= multiplier*metric_func[i][k];
            }
            #pragma unroll
            for (int k { 0 }; k < 4; k++)
            {
                metric_inverse[j][k] -= multiplier*metric_inverse[i][k];
            }
        }
    }

    // Last task is to normalise the rows of metric_inverse by whatever is left in the diagonal of metric_func.
    for (int i { 0 }; i < 4; i++)
    {
        multiplier = 1./metric_func[i][i];
        metric_func[i][i] = 1.;
        #pragma unroll
        for (int k { 0 }; k < 4; k++)
        {
            metric_inverse[i][k] *= multiplier;
        }
    }
}

void getMetricTensor(double x_func[4], double metric_func[4][4])
{
    const double r_s { 1. };
    double r { sqrt(x_func[1]*x_func[1] + x_func[2]*x_func[2] + x_func[3]*x_func[3]) };
    double r_squared { r*r };
    double mult_factor { r_s / (r_squared*(r-r_s)) };
    for (int mu { 1 }; mu < 4; mu++)
    {
        metric_func[0][mu] = 0.;
        metric_func[mu][0] = 0.;
        for (int nu { mu }; nu < 4; nu++)
        {
            metric_func[mu][nu] = mult_factor*x_func[mu]*x_func[nu];
            metric_func[nu][mu] = metric_func[mu][nu];
        }
    }
    metric_func[0][0] = -1. + r_s/r;
    metric_func[1][1] += 1.;
    metric_func[2][2] += 1.;
    metric_func[3][3] += 1.;
}

void getChristoffelSymbols(double x_func[4], double metric_func[4][4], double c_symbols_func[4][4][4], double metric_derivs[4][4][4])
{
    // Assumed default step in each coordinate.
    // TODO: How do you define this adaptively to not break near areas of extreme distortion?
    // For now, just set it to a small number.
    const double step { 1e-4 };
    const double inverse_step { 1./step };

    // Simple Euler forward-difference derivatives of the metric along each component.
    double metric_temp[4][4];
    for (int alpha { 0 }; alpha < 4; alpha++)
    {
        // WARNING: Euler can be significantly faster than central difference; just use a very small step and it'll probably be okay.
        // The limiting factor in accuracy is probably the overarching parameter step in the geodesic equation, not the metric derivatives.
        // TODO: Automatic differentiation with dual numbers?
        x_func[alpha] += step;
        getMetricTensor(x_func, metric_temp);
        for (int mu { 0 }; mu < 4; mu++)
        {
            metric_derivs[alpha][mu][mu] = (metric_temp[mu][mu] - metric_func[mu][mu])*inverse_step;
            for (int nu { mu+1 }; nu < 4; nu++)
            {
                metric_derivs[alpha][mu][nu] = (metric_temp[mu][nu] - metric_func[mu][nu])*inverse_step;
                metric_derivs[alpha][nu][mu] = metric_derivs[alpha][mu][nu];
            }
        }
        x_func[alpha] -= step;
    }

    // Calculate the inverse of metric_func and overwrite it into metric_temp.
    // Set metric_temp to the identity matrix first.
    for (int mu { 0 }; mu < 4; mu++)
    {
        metric_temp[mu][mu] = 1.;
        for (int nu { mu+1 }; nu < 4; nu++)
        {
            metric_temp[mu][nu] = 0.;
            metric_temp[nu][mu] = 0.;
        }
    }
    // Store the inverse metric into metric_temp. metric_func is now useless
    // until it is assigned again in getMetricTensor() (it gets overwritten by invertMetric()).
    invertMetric(metric_func, metric_temp);

    // Calculate the 40 independent Christoffel symbols.
    for (int alpha { 0 }; alpha < 4; alpha++)
    {
        for (int mu { 0 }; mu < 4; mu++)
        {
            for (int nu { mu }; nu < 4; nu++)
            {
                double component[4];
                #pragma unroll
                for (int gamma { 0 }; gamma < 4; gamma++)
                {
                    component[gamma] = metric_derivs[nu][mu][gamma] + metric_derivs[mu][nu][gamma] - metric_derivs[gamma][mu][nu];
                }
                // Remember that metric_temp is the inverse metric here.
                c_symbols_func[alpha][mu][nu] = 0.5*(
                    metric_temp[alpha][0]*component[0] + metric_temp[alpha][1]*component[1]
                    + metric_temp[alpha][2]*component[2] + metric_temp[alpha][3]*component[3]
                );
                c_symbols_func[alpha][nu][mu] = c_symbols_func[alpha][mu][nu];
            }
        }
    }
}

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

    // double x[4] { 0.2, 3., -2., 1. };
    // double metric[4][4];
    // double c_symbols[4][5][4][4][4];
    // double metric_derivs[4][4][4];
    //
    // getMetricTensor(x, metric);
    // getChristoffelSymbols(x, metric, &c_symbols[2][3][0], metric_derivs);
    // for (int i { 0 }; i < 4; i++)
    // {
    //     for (int j { 0 }; j < 4; j++)
    //     {
    //         std::cout << c_symbols[2][3][1][i][j] << "\t";
    //     }
    //     std::cout << "\n";
    // }

    return 0;
}
