#ifndef PARTICLE
#define PARTICLE

#include "world.h"
#include "camera.h"
#include <Eigen/Dense>

/*
 *  Everything in the simulation works in coordinates of (ct, x, y, z)
 *  with the assumption that c = 1 is set, so that coordinates are
 *  in terms of just (t, x, y, z).
 */

using namespace Eigen;

class Particle
{
public:
    // 4-position of the photon in (t, x, y, z) coordinates.
    Vector4d x;
    // 4-velocity of the photon.
    Vector4d v;
    // Metric tensor at the particle's coordinates, assumed to be symmetric (torsion-free).
    Matrix4d metric;
    // Christoffel symbols at the particle's coordinates. These are only used when calculating acceleration
    // and are repeatedly overwritten to avoid allocating and copying; they are never needed elsewhere.
    // One matrix for each coordinate (upper index).
    Matrix4d christoffel_symbols[4] {};
    // dl, the parameter step, will never be larger than this.
    double maxParameterStep { 8. };
    unsigned char* pixelColour { nullptr };

    void setX(Vector4d new_x);
    void setV(Vector4d new_v);
    void makeVNull();
    void normaliseV();
    void updateMetric(Matrix4d new_metric);
    // Advances the simulation using RK4 by a parameter step, dl.
    // TODO: Consider other, potentially more stable ODE integrators, e.g. a symplectic integrator.
    void advance(World &simulation);
    // Returns the scalar product of the 4-velocity; primarily for debugging purposes.
    double scalarProduct();
    // Estimates the deviation of the metric from the Minkowski metric as a cheap way of
    // estimating how twisted spacetime is to adapt the step size. Basically just doing anything
    // to avoid calculating the Riemann tensor.
    double minkowskiDeviation();
    double calculateParameterStep();
    // Calculates the Euclidean distance to the origin; this is useful for some metrics (e.g.
    // calculating if a photon has crossed inside the photon sphere).
    double getEuclideanDistance();
};

// Return the parameter derivative of the given 4-velocity under the current Christoffel symbols.
Vector4d v_derivative(Vector4d v, Matrix4d christoffel_symbols[]);

#endif
