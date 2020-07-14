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
#define DELTA 0.00001
#define MATRIX matrix<double>
#define ROW matrix_row<MATRIX>
#define VECTOR vector<double>
using namespace boost::numeric::ublas;

struct HBETA_RESULT {
    double H;
    VECTOR P;
};

HBETA_RESULT hBeta(VECTOR D, double beta){
    VECTOR P(D.size());
    double H;
    double sumP = 0.0;
    double sumDxP = 0.0;

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

MATRIX x2p(MATRIX points, double tol /*1e-5*/, double perplexity){

    int n = points.size1();
    int d = points.size2();

    VECTOR sum_points(n);

    for(int i=0; i<n; i++){
        sum_points(i) = 0.0;
    }

    for(int i=0; i<n; i++){
        for(int j=0; j<d; j++){
            sum_points(i) += points(i, j) * points(i, j);
        }
    }

    MATRIX D(n, n);
    MATRIX P(n, n);
    VECTOR beta(n);
    double logU = log(perplexity);

    for(int i=0;i<n;i++){
        beta(i) = 1.0;
        for(int j=0;j<n;j++){
            P(i,j) = 0.0;
            D(i,j) = 0.0;     
        }
    }


    MATRIX dotProduct(n,n);
    dotProduct = prod(points, trans(points));

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            D(i,j) = -2 * dotProduct(i,j) + sum_points(i);
        }
    }

    D = trans(D);

    for(int i=0;i<n;i++){
        double sum_pt = sum_points(i);
        for(int j=0;j<n;j++){
            D(i,j) += sum_pt;
        }
    }

    int betamin;
    int betamax;

    for(int i=0; i<n; i++){
        betamin = INT_MIN;
        betamax = INT_MAX;

        ROW rowI(D, i);
        VECTOR Di(rowI);
        remove(Di, i);
        HBETA_RESULT res = hBeta(Di, beta(i));
        double Hdiff = res.H - logU;
        int tries = 0;

        while(abs(Hdiff) > tol && tries < 50){
            if(Hdiff > 0){
                betamin = beta(i);

                if(betamax == INT_MAX || betamax == INT_MIN){
                    beta(i) *= 2;
                } else {
                    beta(i) = (beta(i) + betamax) / 2;
                }
            } else {
                betamax = beta(i);
                if(betamin == INT_MAX || betamin == INT_MIN){
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

    }

    double sumBeta = 0.0;

    for(int i=0; i<n; i++){
        sumBeta += sqrt(1.0 / beta(i));
    }

    std::cout<<"Mean value of sigma: "<< sumBeta/n << std::endl;

    return P;
}

MATRIX refitTSNE(MATRIX points, int stepsNumber, double perplexity, double learning_rate) {
    int n = points.size1();
    
    int max_iter = 1000;
    double initial_momentum = 0.5;
    double final_momentum = 0.8;
    int eta = 500;
    double min_gain = 0.01;
    MATRIX Y(n, 2);
    MATRIX dY(n, 2);
    MATRIX iY(n, 2);
    MATRIX gains(n, 2);
    MATRIX M(n, 2);

    MATRIX probabilities;
    MATRIX P(n, n);

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 1.0);

    //init with random values
    for(int i = 0;i < n;i++) {
        for(int j = 0;j < 2;j++) {
            Y(i, j) = distribution(generator);
            dY(i, j) = 0.0;
            iY(i, j) = 0.0;
            gains(i, j) = 0.0;
        }
    }

    P = x2p(points, 1e-5, perplexity);

    P += trans(P);

    double sumP = 0.0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            sumP += P(i, j);
        }
    }
    P /= sumP;    
    P *= 4;

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            P(i, j) = std::max(P(i, j), 1e-12);
        }
    }

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

        double sumNum = 0.0;

        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                sumNum += num(i,j);
            }
        }

        MATRIX Q = num / sumNum;

        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                Q(i,j) = std::max(Q(i,j), 1e-12);
            }
        }

        // Dobrze do tego momentu

        // Zostalo:

        /*
        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
        */
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

    MATRIX points = readInData("test.csv");
    MATRIX result = refitTSNE(points, 500, 20, 0.1);

    for(int i = 0;i < result.size1();i++) {
        for(int j = 0;j < result.size2();j++) {
            std::cout << result(i, j) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}