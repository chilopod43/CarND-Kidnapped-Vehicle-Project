/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * Set the number of particles. Initialize all particles to 
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1. 
     * Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method 
     *   (and others in this file).
     */
    num_particles = 100;  // Set the number of particles
    std::default_random_engine gen;
    double std_x, std_y, std_theta;  // Standard deviations for x, y, and theta

    // Set standard deviations for x, y, and theta
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2]; 

    // This line creates a normal (Gaussian) distribution for x
    std::normal_distribution<double> dist_x(x, std_x);
    std::normal_distribution<double> dist_y(y, std_y);
    std::normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < num_particles; ++i) {
        double sample_x, sample_y, sample_theta;
        sample_x = dist_x(gen);
        sample_y = dist_y(gen);
        sample_theta = dist_theta(gen);   
        particles.push_back(Particle{i, sample_x, sample_y, sample_theta, 1.0});
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
    /**
     * Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution 
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */
    std::default_random_engine gen;
    std::normal_distribution<double> dist_x(0.0, std_pos[0]);
    std::normal_distribution<double> dist_y(0.0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0.0, std_pos[2]);

    // Calculate new state.
    for (int i=0; i<num_particles; i++) {

        auto& p = particles[i];

        p.x = p.x + velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t ) - sin( p.theta));
        p.y = p.y + velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
        p.theta = p.theta + yaw_rate * delta_t;
    
        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) 
{
    /**
     * Find the predicted measurement that is closest to each 
     *   observed measurement and assign the observed measurement to this 
     *   particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will 
     *   probably find it useful to implement this method and use it as a helper 
     *   during the updateWeights phase.
     */
    // landmarkobs: x,y,id
    for(size_t i=0; i<observations.size(); i++)
    {
        auto& oobs = observations[i];
      
        int tmpid = -1;
        double mind = std::numeric_limits<const double>::infinity();    
        for(size_t j=0; j<predicted.size(); j++)
        {
            auto& pobs = predicted[j];
            double d = dist(oobs.x, oobs.y, pobs.x, pobs.y);
            if(d < mind)
            {
                mind = d;
                tmpid = pobs.id;
            }
        }
        oobs.id = tmpid;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) 
{
    /**
     * Update the weights of each particle using a mult-variate Gaussian 
     *   distribution. You can read more about this distribution here: 
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system. 
     *   Your particles are located according to the MAP'S coordinate system. 
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */
    weights.clear();
    for(int pidx=0; pidx<num_particles; pidx++) 
    {
        auto& p = particles[pidx];

        // -------------------------------------------------------------
        // associate observations with landmarks
        // -------------------------------------------------------------
        std::vector<LandmarkObs> predicted;
        for(size_t lidx=0; lidx<map_landmarks.landmark_list.size(); lidx++) 
        {
            auto& l = map_landmarks.landmark_list[lidx];

            double sensor_dist = dist(p.x, p.y, l.x_f, l.y_f);
            if ( sensor_dist <= sensor_range ) 
            {
                predicted.push_back(LandmarkObs{l.id_i, l.x_f, l.y_f});
            }
        }

        std::vector<LandmarkObs> wld_obses;
        for(size_t oidx=0; oidx<observations.size(); oidx++) 
        {
            auto& o = observations[oidx];
            double wld_ox = p.x + cos(p.theta) * o.x - sin(p.theta) * o.y;
            double wld_oy = p.y + sin(p.theta) * o.x + cos(p.theta) * o.y;
            wld_obses.push_back(LandmarkObs{ o.id, wld_ox, wld_oy});
        }
        dataAssociation(predicted, wld_obses);

        // -------------------------------------------------------------
        // update weights
        // -------------------------------------------------------------
        p.weight = 1.0;
        for(size_t oidx = 0; oidx < wld_obses.size(); oidx++) 
        {
            auto& o = wld_obses[oidx];
          
            LandmarkObs tl;
            for(size_t lidx=0; lidx<predicted.size(); lidx++)
            {
                auto& l = predicted[lidx];
                if(o.id == l.id)
                {
                    tl = l;
                	break;
                }
            }

            double uncertx = std_landmark[0];
            double uncerty = std_landmark[1];
            double coeff = 1 / (2 * M_PI * uncertx * uncerty);
            double xterm = (o.x - tl.x) * (o.x - tl.x) / (2 * uncertx * uncertx);
            double yterm = (o.y - tl.y) * (o.y - tl.y) / (2 * uncerty * uncerty);
            double prob = coeff * exp(-xterm-yterm);
            p.weight *= (prob == 0.0) ? 1e-5 : prob;
        }
        weights.push_back(p.weight);
    }

}

void ParticleFilter::resample() {
    /**
     * Resample particles with replacement with probability proportional 
     *   to their weight. 
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */
    std::default_random_engine gen;
    std::uniform_int_distribution<int> dist_idx(0, num_particles - 1);
    auto maxw = std::max_element(weights.begin(), weights.end());
    std::uniform_real_distribution<double> dist_weight(0.0, *maxw);

    int index = dist_idx(gen);
    double beta = 0.0;

    vector<Particle> resampled;
    for(int i=0; i<num_particles; i++) {
        beta += dist_weight(gen) * 2.0;
        while(beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        resampled.push_back(particles[index]);
    }
    particles = resampled;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
    vector<double> v;

    if (coord == "X") {
      	v = best.sense_x;
    } else {
      	v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}