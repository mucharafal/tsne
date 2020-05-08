#include "Array.hpp"
#include <iostream>
Array::Array(int pointsNumber, int dimensions) {
    points = new double*[pointsNumber];
    for(int i = 0;i < pointsNumber;i++) {
        points[i] = new double[dimensions];
    }
    this->pointsNumber = pointsNumber;
    this->dimensions = dimensions;
}

Array::Array(const Array &obj) {
    points = new double*[obj.pointsNumber];
    for(int i = 0;i < obj.pointsNumber;i++) {
        points[i] = new double[obj.dimensions];
    }
    this->pointsNumber = obj.pointsNumber;
    this->dimensions = obj.dimensions;
    for(int i = 0;i < pointsNumber;i++) {
        for(int j = 0; j < dimensions;j++) {
            (*this)[i][j] = obj[i][j];
        }
    }
}

Array::~Array() {
    for(int i = 0;i < pointsNumber;i++) {
        delete[] points[i];
    }
    delete[] points;
}

int Array::getDimensions() {
    return this->dimensions;
}

int Array::getPointsNumber() {
    return this->pointsNumber;
}

double* Array::getPoint(int i){
    return points[i];
}

double* Array::operator[](int pointNumber) {
    return this->points[pointNumber];
}

const double* Array::operator[](const int pointNumber) const {
    return this->points[pointNumber];
}

Array& Array::operator=(Array other){
    for(int i = 0;i < pointsNumber;i++) {
        delete[] points[i];
    }
    delete[] points;
    
    points = new double*[other.pointsNumber];
    for(int i = 0;i < other.pointsNumber;i++) {
        points[i] = new double[other.dimensions];
    }
    this->pointsNumber = other.pointsNumber;
    this->dimensions = other.dimensions;
    for(int i = 0;i < pointsNumber;i++) {
        for(int j = 0; j < dimensions;j++) {
            (*this)[i][j] = other[i][j];
        }
    }
    return *this;
}