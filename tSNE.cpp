#include <iostream>
#include <fstream>
#include <math.h>
#include <functional>
#include <sstream>
#include <string>
#include <list>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#define DELTA 0.00001
#define MATRIX matrix<double>
#define ROW matrix_row<MATRIX>
#define VECTOR vector<double>
using namespace boost::numeric::ublas;

double calculateEuclideanDistances(ROW a, ROW b) {
    return norm_2(a - b);
}

MATRIX calculateEuclidianDistances(MATRIX points) {
    int pointsNumber = points.size1();
    MATRIX distances(pointsNumber, pointsNumber);

    for(int i=0;i < pointsNumber;i++) {
        distances(i, i) = 0;
        for(int j = i + 1;j < pointsNumber;j++) {;
            double squareDistance = calculateEuclideanDistances(ROW(points, i), ROW(points, j));
            distances(j, i) = distances(i, j) = squareDistance;
        }
    }

    return distances;
}

double pjiFromSigma(int i, int j, MATRIX distances, double sigma) {
    double x = exp(-(distances(i, j) * distances(i, j)) / (2 * sigma * sigma));
    // std::cout << -distances(i, j) / (2 * sigma * sigma) << std::endl;
    double y = 0;
    for(int k = 0;k < distances.size1();k++) {
        for (int l = 0;l < distances.size1();l++) {
            if(k != l) {
                y += exp(-(distances(l, k) * distances(l, k)) / (2 * sigma * sigma));
            }
        }
    }
    // std::cout << "Pji " << x/y << " because x = " << x << " and y = " << y << " sigma: " << sigma << " i: " << i << " j: " << j << " Distance: " << distances(i, j) << std::endl;
    return x/y;
}

double perplexityFromSigma(int i, MATRIX distances, double sigma) {
    double sum = 0;
    for(int j = 0;j < distances.size1();j++) {
        if(j != i) {
            double pji = pjiFromSigma(i, j, distances, sigma);
            sum += pji * log2(pji);
        }
    }
    // std::cout << "Perplexity for i = " << i << ", sigma = " << sigma << " is equal " << pow(2, -sum) << std::endl;
    return pow(2, -sum);
};

double findSigma(int i, MATRIX distances, double perplexity) {
    // This part is a bit complicated. I am not sure, about perplexity function plot
    double start = 0.1;
    double end = 100000;
    double initSearchStep = 0.5;
    bool isGrowing = perplexityFromSigma(i, distances, start) < perplexityFromSigma(i, distances, end);
    if(perplexityFromSigma(i, distances, start) > perplexity && perplexityFromSigma(i, distances, end) > perplexity) {
        std::cout << "Cannot match, too low" << std::endl;
    }

    if(perplexityFromSigma(i, distances, start) < perplexity && perplexityFromSigma(i, distances, end) < perplexity) {
        std::cout << "Cannot match, too big" << std::endl;
    }
    double step = 1;
    double middlePerplexity = 0;
    while((abs((middlePerplexity = perplexityFromSigma(i, distances, ((start + end) / 2))) - perplexity) > DELTA) && (step > DELTA)) {
        double middle = (start + end) / 2;
        if(middlePerplexity < perplexity) {
            if(isGrowing) {
                start = middle;
            } else {
                end = middle;
            }
        } else {
            if(isGrowing) {
                end = middle;
            } else {
                start = middle;
            }
        }
        step = abs(start - end);
        // std::cout << "Step: " << step << "; middlePerplexity = " << middlePerplexity << " perplexity = " << perplexity <<std::endl;
    }
    return (start + end) / 2;
}

MATRIX similaritySNE(MATRIX points, double perplexity) {
    int pointsNumber = points.size1();

    MATRIX p(pointsNumber, pointsNumber);

    const MATRIX distances = calculateEuclidianDistances(points);

    for(int i = 0;i < pointsNumber;i++) {
        double sigma = findSigma(i, distances, perplexity);
        // std::cout << "Sigma[" << i << "]: " << sigma << std::endl;
        for(int j = 0;j < pointsNumber;j++) {
            if(j != i) {
                p(j, i) = pjiFromSigma(i, j, distances, sigma);
            } else {
                p(j, i) = 0;
            } 
        }
    }
    return p;
}

MATRIX symmetrizeProbabilities(MATRIX probabilities) {
    int n = probabilities.size1();

    MATRIX result(n, n);
    // std::cout << "Probabilities: " << std::endl;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            // std::cout << i << " " << j << " " << probabilities(i, j) << " " << probabilities[j][i];
            result(i, j) = (probabilities(i, j) + probabilities(j, i)) / (2 * n);
            // std::cout << " " << result(i, j) << std::endl;
        }
    }

    return result;
}

MATRIX similarityTSNE(MATRIX y) {
    // MATRIX of similarities for lower dimension in tSNE
    int n = y.size1();
    MATRIX distances = calculateEuclidianDistances(y);
    MATRIX q(n, n);

    double denumerator = 0.0;

    for(int k = 0; k < n; k++) {
        for(int l = 0; l < n; l++){
            if(k != l) {
                // std::cout << "Distance: " << distances[k][l] << std::endl;
                denumerator += 1.0 / (1 + distances(k, l) * distances(k, l));
            }
        }
    }
    // std::cout << "Denumerator: " << denumerator << std::endl;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            if(i != j) {
                double numerator = 1.0 / (1 + distances(i, j) * distances(i, j));
                if(numerator == 0) {
                    // std::cout << "Numerator for i: " << i << " and j: " << j << " equal to zero" << std::endl;
                }
                q(i, j) = numerator / denumerator;
            } else {
                q(i, j) = 0;
            }
        }
    }

    return q;
}

MATRIX calculateGradient(MATRIX p, MATRIX q, MATRIX y) {
    int n = y.size1();
    int m = y.size2();
    MATRIX distances = calculateEuclidianDistances(y);

    // std::cout << "Distances calcs; dims: " << distances.size1() << "," << distances.size2() << "\n";

    MATRIX p_minus_q = p - q;

    // std::cout << "p_minus_q calcs; dims: " << p_minus_q.size1() << "," << p_minus_q.size2() << "\n";

    MATRIX gradient(n, m);

    // std::cout << "gradient calcs; dims: " << gradient.size1() << "," << gradient.size2() << "\n";

    for(int i = 0; i < n; i++){   
        for(int j = 0; j < i; j++){
            VECTOR subtracted_vec = ROW (y, i) - ROW(y, j);       // Allocation

            // gradient_i = sum(j=0..n) of (4 * P-Q(i, j) * y[i]-y[j] * (1 + ||y[i] - y[j]||^2)^-1 )
            // below we just perform it element-wise, since y[i] and y[j] are vectors
            for(int k = 0; k < m; k++){
                // std::cout << "Substracted vec " << subtracted_vec[k] << " p-q " << p_minus_q(i, j) << " dist " << distances(i, j) << std::endl;
                subtracted_vec(k) *= 4;
                subtracted_vec(k) *= p_minus_q(i, j);
                subtracted_vec(k) *= 1.0 / (1 + distances(i, j));
                gradient(i, k) = subtracted_vec(k);
            }
        }

        for(int j = i+1; j < n; j++){
            VECTOR subtracted_vec = ROW (y, i) - ROW(y, j);       // Allocation

            // gradient_i = sum(j=0..n) of (4 * P-Q(i, j) * y[i]-y[j] * (1 + ||y[i] - y[j]||^2)^-1 )
            // below we just perform it element-wise, since y[i] and y[j] are vectors
            for(int k = 0; k < m; k++){
                // std::cout << "Substracted vec " << subtracted_vec(k) << " p-q " << p_minus_q(i, j) << " dist " << distances(i, j) << std::endl;
                subtracted_vec(k) *= 4;
                subtracted_vec(k) *= p_minus_q(i, j);
                subtracted_vec(k) *= 1.0 / (1 + distances(i, j));
                gradient(i, k) += subtracted_vec(k);
            }
        }
    }

    return gradient;
}

MATRIX fitTSNE(MATRIX points, int stepsNumber, double perplexity, double learning_rate) {
    // fit lower dimensional space to higher
    int n = points.size1();
    
    double initial_momentum = 0.5;
    double final_momentum = 0.8;
    int eta = 500;
    double min_gain = 0.01;
    MATRIX Y(n, 2);  // = RAND?;
    MATRIX M(n, 2);

    MATRIX probabilities = similaritySNE(points, perplexity);

    MATRIX P = symmetrizeProbabilities(probabilities);

    //init with random values
    for(int i = 0;i < n;i++) {
        for(int j = 0;j < 2;j++) {
            Y(i, j) = 0.1 * rand() / RAND_MAX;
            M(i, j) = 0;  
        }
    }

    

    for(int step = 0; step < stepsNumber; step++){
        // std::cout << "Step: " << step << "\n";
        MATRIX Q = similarityTSNE(Y);
        // std::cout << "Q calcs\n";
        MATRIX gradient = calculateGradient(P, Q, Y);
        // std::cout << "Gradient calcs\n";
        //   Y_1 = Y + (momentum * M) + (learning_rate * gradient);
        MATRIX Y_1(Y.size1(), Y.size2());
        double momentum = step < 20 ? initial_momentum : final_momentum;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < 2; j++){
                Y_1(i, j) = Y(i, j) + (learning_rate * gradient(i, j)) + (momentum * M(i, j));
                // std::cout<<"Step: " << (learning_rate * gradient(i, j)) + (momentum * M(i, j)) << "\n";
            }
        }
        //   M = Y_1 - Y;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < 2; j++){
                M(i, j) = Y_1(i, j) - Y(i, j);
            }
        }

        Y = Y_1;

        // Print iteration error
        if ((step + 1) % 10 == 0){
            double C = 0.0;

            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    if(i != j) {
                        // std::cout << "P[" << i << "][" << j << "] = " << P(i, j) << " Q: " << Q(i, j) << std::endl; 
                        C += P(i, j) * log(P(i, j) / Q(i, j));
                    }
                }
            }

            std::cout << "Iteration " << step + 1 << ": error is " << C << std::endl;
        }
    }

    return Y;
}

MATRIX readInData(std::string csv_filename){
    using namespace std;
    // Load points vectors from file

    std::ifstream file;
    file.open(csv_filename);

    int dimensions = 0;

    if(file.good()){
        string header_line;
        getline(file, header_line, '\n');

        istringstream header_iss(header_line);
        string column_name;

        while (getline(header_iss, column_name, ',')){
            dimensions++;
        }
    }

    list<double*> points;

    while(file.good()){
        string point_line;
        getline(file, point_line, '\n');

        if(point_line.empty()){
            // Skip empty lines
            continue;
        }

        istringstream iss(point_line);
        string val;

        double* point = new double[dimensions];
        int i = 0;

        // Split line using ',' as delimiter

        while (getline(iss, val, ',')){
            point[i++] = stod(val);
        }

        points.push_back(point);
    }
    file.close();

    // Load points vectors into MATRIX

    MATRIX result(points.size(), dimensions);

    list<double*>::iterator it = points.begin();
    for (int i = 0; i < points.size(); i++){
        for(int j = 0; j < dimensions; j++){
            result(i, j) = (*it)[j];
        }
        
        it++;
    }

    return result;
}


int main() {

    MATRIX points = readInData("processed_mnist.csv");
    MATRIX result = fitTSNE(points, 1000, 30, 200);

    for(int i = 0;i < result.size1();i++) {
        for(int j = 0;j < result.size2();j++) {
            std::cout << result(i, j) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}