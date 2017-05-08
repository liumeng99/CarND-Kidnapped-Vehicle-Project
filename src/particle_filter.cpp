/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.

    num_particles = 100;

    std::default_random_engine generator;
    std::normal_distribution<double> x_d(x, std[0]);
    std::normal_distribution<double> y_d(y, std[1]);
    std::normal_distribution<double> theta_d(theta, std[2]);
    for (int i=0; i<num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = x_d (generator);
        p.y = y_d (generator);
        p.theta = theta_d (generator);
        p.weight = 1;
        particles.push_back(p);
        weights.push_back(1);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // predict particles states and add random Gaussian noise.
    std::default_random_engine generator;
    std::normal_distribution<double> x_d(0, std_pos[0]);
    std::normal_distribution<double> y_d(0, std_pos[1]);
    std::normal_distribution<double> theta_d(0, std_pos[2]);

    for (auto& p : particles) {
        double new_x = p.x;
        double new_y = p.y;
        double new_theta = p.theta;
        //avoid division by zero
        if (fabs(yaw_rate) < 1e-2) {
            new_x += delta_t * velocity * cos (p.theta);
            new_y += delta_t * velocity * sin (p.theta);
        } else {
            new_x += velocity / yaw_rate * (sin (p.theta + yaw_rate * delta_t) - sin (p.theta));
            new_y += velocity / yaw_rate * (- cos (p.theta + yaw_rate * delta_t) + cos (p.theta));
        }
        p.x = new_x + x_d(generator);
        p.y = new_y + y_d(generator);
        p.theta += delta_t * yaw_rate + theta_d(generator);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    //not used
}

/**
 * @brief convert observation to mao coordinates
 * @param p - particle that observed landmark
 * @param l_in_local_coords - landmark in local coordinates
 * @param l_in_global_coords - output, landmark in map coordinates
 */
void convertObservationToGlobalCoords (const Particle& p, const LandmarkObs& l_in_local_coords, LandmarkObs& l_in_global_coords) {
    double s = sin (p.theta);
    double c = cos (p.theta);
    l_in_global_coords.x = l_in_local_coords.x * c - l_in_local_coords.y * s + p.x;
    l_in_global_coords.y = l_in_local_coords.x * s + l_in_local_coords.y * c + p.y;
}

// if you need only comparsion of distances you probably don't need sqrt
double dist2 (const Map::single_landmark_s& a, const LandmarkObs& b) {
    return (a.x_f-b.x)*(a.x_f-b.x) + (a.y_f-b.y)*(a.y_f-b.y);
}

double dist (const Map::single_landmark_s& l, const Particle& p) {
    return std::sqrt ((l.x_f-p.x)*(l.x_f-p.x) + (l.y_f-p.y)*(l.y_f-p.y));
}

double dist (const Map::single_landmark_s& a, const LandmarkObs& b) {
    return std::sqrt (dist2(a, b));
}

/**
 * @brief used to sort map landmarks to find the closest landmark to observation
 */
struct sort_by_distance {
    sort_by_distance (const LandmarkObs& ref) : ref(ref) {}
    const LandmarkObs& ref;

    bool operator() (const Map::single_landmark_s& a,const Map::single_landmark_s& b) {
        return (dist2(a, ref) < dist2(b, ref));
    }
};

/**
 * @brief associate observed observations with map landmarks, finds closest neighbours
 * @param map_landmarks
 * @param observations
 * @param associated_landmarks
 */
void associateObservations (
    const std::vector<Map::single_landmark_s>& map_landmarks,
    std::vector<LandmarkObs>& observations,
    std::vector<Map::single_landmark_s>& associated_landmarks
) {
    associated_landmarks.clear();
    std::vector<Map::single_landmark_s> landmarks = map_landmarks;
    for (auto& o : observations) {
        sort_by_distance s (o);
        //sort by distance
        std::sort (landmarks.begin(), landmarks.end(), s);
        //get closest
        associated_landmarks.push_back(landmarks.front());
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

    //if no observations just return
    if (observations.size() < 1) return;

    double norm = 1.0 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    double coeff1 = 1.0 / (2 * std_landmark [0] * std_landmark [0]);
    double coeff2 = 1.0 / (2 * std_landmark [1] * std_landmark [1]);

    for (int pi=0; pi<num_particles; pi++) {

        auto& p = particles [pi];

        //getting map landmarks that possibly could be spotted
        std::vector <Map::single_landmark_s> visible_landmarks;
        for (auto& l : map_landmarks.landmark_list) {
            //need 1.5x gap to properly handle particles at the edge of sensor range
            if (dist(l, p) <= 1.5 * sensor_range) {
                visible_landmarks.push_back(l);
            }
        }

        //if no possible visible observations
        //continue and leave weight untouched
        if (visible_landmarks.size() < 1) {
            continue;
        }

        //converting observations into map coordinates
        std::vector<LandmarkObs> obs_on_map;
        for (auto& o : observations) {
            LandmarkObs o_in_global;
            convertObservationToGlobalCoords(p, o, o_in_global);
            obs_on_map.push_back(o_in_global);
        }
        std::vector<Map::single_landmark_s> associated_landmarks;

        //associate observations and landmarks
        associateObservations(visible_landmarks, obs_on_map, associated_landmarks);

        //calculate particles weights
        double w = 1;
        for (int i=0; i<obs_on_map.size(); i++) {
            auto& o = obs_on_map [i];
            auto& l = associated_landmarks [i];

            w *= norm * exp (-(coeff1 * (o.x - l.x_f) * (o.x - l.x_f) + coeff2 * (o.y - l.y_f) * (o.y - l.y_f)));
        }
        weights[pi] = w;
    }
}

void ParticleFilter::resample() {
    std::vector<Particle> new_particles;
    std::default_random_engine generator;

    std::discrete_distribution<int> d (weights.begin(), weights.end());
    for (int i=0; i<num_particles; i++) {
        int ind = d (generator);
        new_particles.push_back(particles[ind]);
    }
    particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
