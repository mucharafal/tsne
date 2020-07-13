class Array {
    private:
    double** points;
    int pointsNumber;
    int dimensions;

    public:

    Array(int pointsNumber, int dimensions);

    Array(const Array &obj);

    Array& operator=(Array other);

    ~Array();

    int getDimensions();

    int getPointsNumber();

    double* getPoint(int i);

    double* operator[](int pointNumber);

    const double* operator[](const int pointNumber) const;
};