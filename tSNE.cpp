#include <iostream>
#include <math.h>
#include <functional>
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

Array similarityTSNE(...) {
    // array of similarities for lower dimension in tSNE
}

double* calculateGradient(...) {
    //calculate gradient for lower dimension
}

Array fitTSNE(Array points, int stepsNumber, double perplexity) {
    // fit lower dimensional space to higher
}

int main() {

    Array points = readInData();

    fitTSNE(points);

    cout << "tSNE!" << endl;

    return 0;
}