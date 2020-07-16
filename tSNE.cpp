// Basing on python implementation by Laurens van der Maaten on 20-12-08 which allows non-comercial usage.
// Authors: Grzegorz Frejek, Rafał Mucha

#include <iostream>
#include <fstream>
#include <math.h>
#include <functional>
#include <sstream>
#include <string>
#include <list>
#include <random>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/range/algorithm.hpp>
#include <chrono> 
using namespace std::chrono; 
#define DELTA 0.00001
#define MATRIX matrix<long double>
#define ROW matrix_row<MATRIX>
#define COLUMN matrix_column<MATRIX>
#define VECTOR vector<long double>
#define MATRIX_OF scalar_matrix<long double>
#define NDEBUG 1
#define BOOST_UBLAS_USE_LONG_DOUBLE 1
using namespace boost::numeric::ublas;

auto app_start = high_resolution_clock::now(); 

MATRIX tile(VECTOR row, int numberOfRows) {
    MATRIX res(numberOfRows, row.size());
    for(int i = 0;i < numberOfRows;i++) {
        ROW(res, i) = row;
    }
    return res;
}

VECTOR sum(MATRIX input_m, int axis) {
    MATRIX m = input_m;
    if(axis == 1) {
        m = trans(input_m);
    } 
    int columns_number = m.size2();
    VECTOR out(columns_number);
    for(int i = 0;i < columns_number;i++) {
        out(i) = sum(COLUMN(m, i));
    }
    return out;
}

VECTOR mean(MATRIX input_m, int axis) {
    VECTOR sums = sum(input_m, axis);
    if(axis == 1) {
        return sums / input_m.size2();
    } else {
        return sums / input_m.size1();
    }
}

struct HBETA_RESULT {
    long double H;
    VECTOR P;
};

HBETA_RESULT hBeta(VECTOR D, long double beta){
    VECTOR P(D.size());
    long double H;
    long double sumP = 0.0;
    long double sumDxP = 0.0;

    for(int i=0; i < D.size(); i++){
        P(i) = exp(-D(i) * beta);
        sumP += P(i);
        sumDxP += P(i) * D(i);
    }

    H = log(sumP) + beta * sumDxP / sumP;

    P /= sumP;

    return HBETA_RESULT{
        H,
        P
    };
}

template<class T> void remove(vector<T> &v, uint idx)
{
    assert(idx < v.size());
    for (uint i = idx; i < v.size() - 1; i++) {
        v[i] = v[i + 1];
    }
    v.resize(v.size() - 1);
}

MATRIX x2p(MATRIX X, long double tol /*1e-5*/, long double perplexity){

    int n = X.size1();
    int d = X.size2();

    VECTOR sum_X = sum(element_prod(X, X), 1);

    MATRIX D(n, n);
    MATRIX P(n, n);
    VECTOR beta(n);
    long double logU = log(perplexity);

    for(int i=0;i<n;i++){
        beta(i) = 1.0;
        for(int j=0;j<n;j++){
            P(i,j) = 0.0;
            D(i,j) = 0.0;     
        }
    }

    MATRIX dotProduct = prod(X, trans(X));

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            D(i,j) = -2 * dotProduct(i,j) + sum_X(i);
        }
    }

    D = trans(D);

    for(int i=0;i<n;i++){
        long double sum_pt = sum_X(i);
        for(int j=0;j<n;j++){
            D(i,j) += sum_pt;
        }
    }

    double betamin;
    double betamax;

    for(int i=0; i<n; i++){
        betamin = -__DBL_MAX__;
        betamax = __DBL_MAX__;

        VECTOR Di = ROW (D, i);
        remove(Di, i);
        HBETA_RESULT res = hBeta(Di, beta(i));
        long double Hdiff = res.H - logU;
        int tries = 0;

        while(abs(Hdiff) > tol && tries < 50){
            if(Hdiff > 0){
                betamin = beta(i);

                if(betamax == __DBL_MAX__ || betamax == -__DBL_MAX__){
                    beta(i) *= 2;
                } else {
                    beta(i) = (beta(i) + betamax) / 2;
                }
            } else {
                betamax = beta(i);
                if(betamin == __DBL_MAX__ || betamin == -__DBL_MAX__){
                    beta(i) /= 2;
                } else {
                    beta(i) = (beta(i) + betamin) / 2;
                }
            }

            res = hBeta(Di, beta(i));
            Hdiff = res.H - logU;
            tries += 1;
        }

        for(int j = 0; j < n; j++){
            if(j != i){
                int index = j;
                if(index > i){
                    index--;
                }
                P(i, j) = res.P(index);
            }
        }

        if((i + 1) % 500 == 0) {

            auto duration = duration_cast<seconds>(high_resolution_clock::now() - app_start);
            std::cerr << "Processed " << i + 1 << " of " << n << "time from app start: " << duration.count() << " seconds\n";
        }
    }

    long double sumBeta = 0.0;

    for(int i=0; i<n; i++){
        sumBeta += sqrt(1.0 / beta(i));
    }

    std::cerr<<"Mean value of sigma: "<< sumBeta/n << std::endl;

    return P;
}



MATRIX refitTSNE(MATRIX points, int stepsNumber, long double perplexity, long double learning_rate) {
    int n = points.size1();
    int no_dims = 2;  

    int max_iter = stepsNumber;
    long double initial_momentum = 0.5;
    long double final_momentum = 0.8;
    int eta = 500;
    long double min_gain = 0.01;
    MATRIX Y(n, 2);
    MATRIX dY(n, 2);
    MATRIX iY(n, 2);
    MATRIX gains(n, 2);
    MATRIX M(n, 2);

    MATRIX probabilities;
    MATRIX P(n, n);

    std::default_random_engine generator;
    std::normal_distribution<long double> distribution(0, 1e-4);

    //init with random values
    for(int i = 0;i < n;i++) {
        for(int j = 0;j < 2;j++) {
            Y(i, j) = distribution(generator);
            dY(i, j) = 0.0;
            iY(i, j) = 0.0;
            gains(i, j) = 1.0;
        }
    }

    P = x2p(points, 1e-5, perplexity);

    P += trans(P);

    long double sumP = 0.0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            sumP += P(i, j);
        }
    }
    P /= sumP;    
    P *= 4;

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            P(i, j) = std::max(P(i, j), (long double)1e-12);
        }
    }

    auto duration = duration_cast<seconds>(high_resolution_clock::now() - app_start);
    std::cerr << "P calculated; time from start app: " << duration.count() << " seconds\n";

    for(int iter = 0; iter < max_iter; iter++){
        VECTOR sum_Y(n);
        for(int i = 0; i < n; i++){
            sum_Y(i) = Y(i, 0) * Y(i, 0) + Y(i, 1) * Y(i, 1);
        }

        MATRIX num = -2.0 * prod(Y, trans(Y));
        
        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                if(i == j){
                    num(i,j) = 0.0;
                } else {
                    num(i,j) = 1.0 / (1.0 + num(i, j) + sum_Y(i) + sum_Y(j));
                }
            }
        }
        num = trans(num);

        long double sumNum = 0.0;

        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                sumNum += num(i,j);
            }
        }

        MATRIX Q = num / sumNum;

        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                Q(i,j) = std::max(Q(i,j), (long double)1e-12);
            }
        }

        // Compute gradient
        MATRIX PQ = P - Q;
        for(int i = 0;i < n;i++) {
            VECTOR PQi = COLUMN(PQ, i);
            VECTOR NUMi = COLUMN(num, i);
            VECTOR row = element_prod(PQi, NUMi);
            MATRIX tileRes = tile(row, no_dims);
            MATRIX subY = tile(ROW(Y, i), n) - Y;
            MATRIX toSum = element_prod(trans(tileRes), subY);
            ROW(dY, i) = sum(toSum, 0);
        }
        
        //# Perform the update
        long double momentum = iter < 20 ? initial_momentum : final_momentum;
        for(int i = 0;i < n;i++) {
            for(int j = 0;j < no_dims;j++) {
                if(dY(i, j) > 0. == iY(i, j) > 0.) {
                    gains(i, j) *= 0.8;
                } else {
                    gains(i, j) += 0.2;
                }
                gains(i, j) = std::max(gains(i, j), min_gain);
            }
        }

        iY = momentum * iY - eta * element_prod(gains, dY);  

        Y = Y + iY;
        Y = Y - tile(mean(Y, 0), n);
        
        // # Compute current value of cost function
        if((iter + 1) % 10 == 0) {
            long double C = 0.0;

            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    if(i != j) {
                        C += P(i, j) * log(P(i, j) / Q(i, j));
                    }
                }
            }
            if(C < 1e-5) {
                return Y;
            }
            auto duration = duration_cast<seconds>(high_resolution_clock::now() - app_start);
            std::cerr << "Iteration " << iter + 1 << ": error is " << C << " time from start app: " << duration.count() << " seconds\n";
        }
        // # Stop lying about P-values
        if(iter == 100) {
            P = P / 4.;
        }

        auto duration = duration_cast<seconds>(high_resolution_clock::now() - app_start);
        std::cerr << "After iteration: " << iter << "; time from start app: " << duration.count() << " seconds" << std::endl;
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

    list<long double*> points;

    while(file.good()){
        string point_line;
        getline(file, point_line, '\n');

        if(point_line.empty()){
            // Skip empty lines
            continue;
        }

        istringstream iss(point_line);
        string val;

        long double* point = new long double[dimensions];
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

    list<long double*>::iterator it = points.begin();
    for (int i = 0; i < points.size(); i++){
        for(int j = 0; j < dimensions; j++){
            result(i, j) = (*it)[j];
        }
        
        it++;
    }

    return result;
}

void normalize(MATRIX &a) {
    long double minValue = __DBL_MAX__;
    long double maxValue = -__DBL_MAX__;
    int n = a.size1(), d = a.size2();
    for(int i = 0;i < n;i++) {
        for(int j = 0;j < d;j++) {
            minValue = std::min(a(i, j), minValue);
            maxValue = std::max(a(i, j), maxValue);
        }
    }

    long double scale = maxValue - minValue;

    for(int i = 0;i < n;i++) {
        for(int j = 0;j < d;j++) {
            a(i, j) = (a(i, j) - minValue) / scale;
        }
    }
}

int main() {

    MATRIX points = readInData("mnist_normal_5k_pca30.csv");
    normalize(points);
    MATRIX result = refitTSNE(points, 500, 50, 0.1);


    for(int i = 0;i < result.size1();i++) {
        for(int j = 0;j < result.size2();j++) {
            std::cout << result(i, j) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}