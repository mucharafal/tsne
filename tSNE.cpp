#include <iostream>
#include <fstream>
#include <math.h>
#include <functional>
#include <sstream>
#include <string>
#include <list>
#define DELTA 0.00001

using namespace std;

class Array {
    private:
    double** points;
    int pointsNumber;
    int dimensions;

    public:

    Array(int pointsNumber, int dimensions) {
        points = new double*[pointsNumber];
        for(int i = 0;i < pointsNumber;i++) {
            points[i] = new double[dimensions];
        }
        this->pointsNumber = pointsNumber;
        this->dimensions = dimensions;
    }

    ~Array() {
        for(int i = 0;i < pointsNumber;i++) {
            delete points[dimensions];
        }
        delete points;
    }

    int getDimensions() {
        return this->dimensions;
    }

    int getPointsNumber() {
        return this->pointsNumber;
    }

    double* getPoint(int i){
        return points[i];
    }

    double* operator[](int pointNumber) {
        return this->points[pointNumber];
    }

    const double* operator[](const int pointNumber) const {
        return this->points[pointNumber];
    }
};

double* substractVectors(double* a, double* b, int size) {  // Warn! Allocate memory!
    double* result = new double[size];
    for(int i = 0;i < size;i++) {
        result[i] = a[i] - b[i];
    }
    return result;
}

double calculateEuclideanDistances(double *a, double *b, int size) {
    double *substracted = substractVectors(a, b, size);
    double result = 0;
    for(int i = 0;i < size;i++) {
        result += substracted[i] * substracted[i];
    }
    delete substracted;
    return result;
}

Array calculateSquareEuclidianDistances(Array points) {
    int pointsNumber = points.getPointsNumber();
    Array distances(pointsNumber, pointsNumber);

    for(int i=0;i < pointsNumber;i++) {
        distances[i][i] = 0;
        for(int j = i + 1;j < pointsNumber;j++) {
            double squareDistance = calculateEuclideanDistances(points[i], points[j], pointsNumber);
            distances[i][j] = distances[j][i] = squareDistance;
        }
    }
    return distances;
}

double findSigma(function<double(double)> perplexityFromSigma, double perplexity) {
    // This part is a bit complicated. I am not sure, about perplexity function plot
    double start = 0.0001;
    double end = 100;
    double initSearchStep = 0.5;
    bool isGrowing = perplexityFromSigma(start) < perplexityFromSigma(end);
    while(perplexityFromSigma(((start + end) / 2 - perplexity) > DELTA)) {
        double middle = (start + end) / 2;
        double perplexityForMiddle = perplexityFromSigma(middle);
        if(perplexityForMiddle < perplexity) {
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
    }
    return (start + end) / 2;
}

Array similaritySNE(Array points, double perplexity) {
    int pointsNumber = points.getPointsNumber();

    Array p(pointsNumber, pointsNumber);
    Array distances = calculateSquareEuclidianDistances(points);

    for(int i = 0;i < pointsNumber;i++) {
        function <double(double)> *pjis = new function<double(double)>[pointsNumber];
        for(int j = 0;j < pointsNumber;j++) {
            auto pjiFromSigma = [distances, i, j, pointsNumber] (double sigma) -> double {
                double x = exp(-distances[i][j] / (2 * sigma * sigma));
                double y = 0;
                for(int k = 0;k < pointsNumber;k++) {
                    if(k != i) {
                        y += exp(-distances[i][k] / (2 * sigma * sigma));
                    }
                }
                return x/y;
            };
            pjis[j] = pjiFromSigma;
        }
        auto perplexityFromSigma = [pjis, pointsNumber](double sigma) -> double {
            double sum = 0;
            for(int i = 0;i < pointsNumber;i++) {
                double pji = pjis[i](sigma);
                sum += pji * log2(pji);
            }
            return -sum;
        }; 
        double sigma = findSigma(perplexityFromSigma, perplexity);

        for(int j = 0;j < pointsNumber;j++) {
            if(j != i) {
                p[j][i] = pjis[j](sigma);
            } else {
                p[j][i] = 0;
            } 
        }
        delete pjis;
    }
    return p;
}

Array symmetrizeProbabilities(Array probabilities) {
    int n = probabilities.getPointsNumber();

    Array result(n, n);

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            result[i][j] = (probabilities[i][j] + probabilities[j][i]) / 2 * n;
        }
    }

    return result;
}

Array similarityTSNE(Array y) {
    // array of similarities for lower dimension in tSNE
    int n = y.getPointsNumber();
    Array distances = calculateSquareEuclidianDistances(y);
    Array q(n, n);

    double denumerator = 0.0;

    for(int k = 0; k < n; k++) {
        for(int l = 0; l < n; l++){
            if(k != l) {
                denumerator += 1 / (1 + distances[k][l]);
            }
        }
    }

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            double numerator = 1 / (1 + distances[i][j]);
            q[i][j] = numerator / denumerator;
        }
    }

    return q;
}

double** calculateGradient(Array p, Array q, Array y) {
    int n = p.getPointsNumber();
    Array distances = calculateSquareEuclidianDistances(y);
    Array p_minus_q(n, n);

    double** gradient = new double*[n];             // Allocation

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            p_minus_q[i][j] = p[i][j] - q[i][j];
        }
    }

    for(int i = 0; i < n; i++){
        double gradient_i[n] = { 0.0 };
        for(int j = 0; j < n; j++){
            double* subtracted_vec = substractVectors(y[i], y[j], n);       // Allocation

            // gradient_i = sum(j=0..n) of (4 * P-Q[i][j] * y[i]-y[j] * (1 + ||y[i] - y[j]||^2)^-1 )
            // below we just perform it element-wise, since y[i] and y[j] are vectors
            for(int k = 0; k < n; k++){
                subtracted_vec[k] *= 4;
                subtracted_vec[k] *= p_minus_q[i][j];
                subtracted_vec[k] *= 1 / (1 + distances[i][j]);
                gradient_i[k] += subtracted_vec[k];
            }

            delete subtracted_vec;
        }

        gradient[i] = gradient_i;
    }

    return gradient;
}

Array fitTSNE(Array points, int stepsNumber, double perplexity, double learning_rate) {
    // fit lower dimensional space to higher
    int n = points.getPointsNumber();
    
    double initial_momentum = 0.5;
    double final_momentum = 0.8;
    int eta = 500;
    double min_gain = 0.01;
    Array Y(n, 2);  // = RAND?;
    Array M(n, 2);

    cout << "Probabilities" << endl;
    Array probabilities = similaritySNE(points, perplexity);
    cout << "Symmetrize" << endl;
    Array P = symmetrizeProbabilities(probabilities);


    for(int step = 0; step < stepsNumber; step++){
        cout << "Step: " << step + 1 << endl;

        Array Q = similarityTSNE(Y);
        double** gradient = calculateGradient(P, Q, Y);

        //   Y_1 = Y + (momentum * M) + (learning_rate * gradient);
        Array Y_1(Y.getPointsNumber(), Y.getDimensions());
        double momentum = step < 20 ? initial_momentum : final_momentum;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                Y_1[i][j] = Y[i][j] + (learning_rate * gradient[i][j]) + (momentum * M[i][j]);
            }
        }

        //   M = Y_1 - Y;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                M[i][j] = Y_1[i][j] - Y[i][j];
            }
        }

        Y = Y_1;


        // Print iteration error
        if ((step + 1) % 10 == 0){
            double C = 0.0;

            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    C += P[i][j] * log(P[i][j] / Q[i][j]);
                }
            }

            cout << "Iteration " << step + 1 << ": error is " << C << endl;
        }
    }

    return Y;
}

Array readInData(string csv_filename){
    // Load points vectors from file

    ifstream file;
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

    // Load points vectors into Array

    Array result(points.size(), dimensions);

    list<double*>::iterator it = points.begin();
    for (int i = 0; i < points.size(); i++){
        for(int j = 0; j < dimensions; j++){
            result[i][j] = (*it)[j];
        }
        
        it++;
    }

    return result;
}


int main() {

    Array points = readInData("test.csv");
    fitTSNE(points, 500, 30, 0.5);

    return 0;
}