#include "particle.h"
#include "world.h"
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;

void Particle::setX(Vector4d new_x)
{
    x = new_x;
}

void Particle::setV(Vector4d new_v)
{
    v = new_v;
}

/*
 * Modifies the t-component of the 4-velocity to make the vector null.
 * R equires v and the* metric at the current coordinates to be defined.
 * This requires solving a quadratic equation for the t-component; assume
 * that you should take the more positive component because g_00 is
 * probably negative. Note that both solutions are negative in a black hole.
 * The raytracer evolves photons "backwards", as if they are being emitted
 * from the camera/observer. Use this when defining the initial velocities
 * of photons to force them to be null.
 */
void Particle::makeVNull()
{
    Vector3d spatial_v { v(seq(1, 3)) };
    double a { metric(0, 0) };
    double b { 2.*metric(seq(1, 3), 0).dot(spatial_v) };
    Matrix3d spatial_metric { metric(seq(1, 3), seq(1, 3)) };
    double c { spatial_v.dot(spatial_metric*spatial_v) };
    // Want the solution going backwards in time since we are tracing in reverse.
    // Doesn't actually matter whether you choose + or -.
    v(0) = (-b - sqrt(b*b - 4.*a*c)) / (2.*a);
    normaliseV();
}

// Makes the L2 (Euclidean) norm of the 4-velocity 1 for the sake
// of maintaining a roughly consistent affine parameterisation.
// WARNING: this is not a "unit" velocity!
void Particle::normaliseV()
{
    v /= v.norm();
}

void Particle::updateMetric(Matrix4d new_metric)
{
    metric = new_metric;
}

// Advances the ray path by a parameter step, dl.
// WARNING: Due to the null constraint, there are only three independent
// velocity components, but it's much simpler to integrate all 4 (and
// probably not slower or any less accurate).
// TODO: Adapt the step-size based on how far the metric at the position
// deviates from the Minkowski metric.
void Particle::advance(World &simulation)
{
    // Calculate parameter step.
    double dl { calculateParameterStep() };

    // Currently advances with RK4.
    Vector4d x_step { 0., 0., 0., 0. };
    Vector4d v_step { 0., 0., 0., 0. };

    Vector4d k_n_minus_1_x;
    Vector4d k_n_x;
    Vector4d k_n_minus_1_v;
    Vector4d k_n_v;

    Vector4d temp_x;
    Vector4d temp_v;
    Matrix4d temp_metric;

    // Calculate k_1.
    simulation.getChristoffelSymbols(x, metric, christoffel_symbols);
    k_n_x = v;
    k_n_v = v_derivative(v, christoffel_symbols);
    x_step += k_n_x;
    v_step += k_n_v;

    // Calculate k_2 and k_3.
    for (int i { 0 }; i < 2; i++)
    {
        k_n_minus_1_x = k_n_x;
        k_n_minus_1_v = k_n_v;
        temp_x = x + 0.5*dl*k_n_minus_1_x;
        temp_v = v + 0.5*dl*k_n_minus_1_v;
        temp_metric = simulation.getMetricTensor(temp_x);
        simulation.getChristoffelSymbols(temp_x, temp_metric, christoffel_symbols);
        k_n_x = v + 0.5*dl*k_n_minus_1_v;
        k_n_v = v_derivative(temp_v, christoffel_symbols);
        x_step += 2.*k_n_x;
        v_step += 2.*k_n_v;
    }

    // Calculate k_4.
    k_n_minus_1_x = k_n_x;
    k_n_minus_1_v = k_n_v;
    temp_x = x + dl*k_n_minus_1_x;
    temp_v = v + dl*k_n_minus_1_v;
    temp_metric = simulation.getMetricTensor(temp_x);
    simulation.getChristoffelSymbols(temp_x, temp_metric, christoffel_symbols);
    k_n_x = v + dl*k_n_minus_1_v;
    k_n_v = v_derivative(temp_v, christoffel_symbols);
    x_step += k_n_x;
    v_step += k_n_v;

    // Advance x and v.
    x += (dl/6.)*x_step;
    v += (dl/6.)*v_step;

    updateMetric(simulation.getMetricTensor(x));
}

// Scalar product of the 4-velocity. Primarily for debugging to check the scalar product is conserved.
double Particle::scalarProduct()
{
    return v.dot(metric*v);
}

// Tries to estimate how far the metric deviates from the Minkowski metric as a cheap way
// of estimating how curved the spacetime is without resorting to the Riemann tensor.
double Particle::minkowskiDeviation()
{
    // Try to "normalise" against things that scale the entire metric
    // but don't actually cause curvature.
    double scale_factor { metric.diagonal().cwiseAbs().sum() / 4. };
    Matrix4d minkowski;
    minkowski.setIdentity();
    minkowski(0, 0) = -1.;
    Matrix4d deviation { metric/scale_factor - minkowski };
    // Sum up the absolute values of this deviation matrix.
    return deviation.cwiseAbs().sum();
}

// Calculates an adaptive step size by estimating how curved the space is.
double Particle::calculateParameterStep()
{
    // Designate a deviation of approximately 3 to give a step size of 0.02 (this produces good stability
    // for many orbits in the photon sphere around a Schwarzschild black hole of r_s = 1).
    double deviation { minkowskiDeviation() };
    if (deviation == 0.)
    {
        // Exactly the Minkowski metric.
        return maxParameterStep;
    }
    double step { 1e-1 * pow(3./deviation, 2.) };
    if (step < maxParameterStep)
    {
        return step;
    }
    else
    {
        return maxParameterStep;
    }
}

// Calculates the Euclidean "distance"; this is useful for some metrics (e.g.
// calculating if a photon has crossed inside the photon sphere of the Schwarzschild metric)
// and for checking if a photon has escaped sufficiently far to infinity (i.e. hit the
// background skysphere/sky map).
double Particle::getEuclideanDistance()
{
    return x(seq(1, 3)).norm();
}

// Returns the derivative of the given 4-velocity using the current Christoffel symbols.
Vector4d v_derivative(Vector4d v, Matrix4d christoffel_symbols[])
{
    Vector4d acceleration;
    for (int mu { 0 }; mu < 4; mu++)
    {
        acceleration(mu) = -1.*v.dot(christoffel_symbols[mu]*v);
    }

    return acceleration;
}
