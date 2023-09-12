#include <stdio.h>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>

#include "ubyte-reader.h"

struct Range {
    float b[2];
    float w[2];
};

// Activation functions
using fn = std::vector<float>(const std::vector<float>&);
std::vector<float> bypass(const std::vector<float>& v);
std::vector<float> tanh(const std::vector<float>& v);
std::vector<float> softmax(const std::vector<float>& v);
// Derivatives of the activation funcs, used for gradient descend
// input m-sized vector, output m * m matrix
using dfn = std::vector<std::vector<float>>(const std::vector<float>&);
std::vector<std::vector<float>> dBypass(const std::vector<float>& v);
std::vector<std::vector<float>> dTanh(const std::vector<float>& v);
std::vector<std::vector<float>> dSoftmax(const std::vector<float>& v);

class DataSet {
   public:
    uchar** images;
    uchar* labels;
    int num;
    int dimX;
    int dimY;

    int getImgSize() const { return dimX * dimY; }

    uchar* getImageAt(int idx) const {
        if (idx >= num || idx < 0) throw std::runtime_error("Index out of bound!");
        return images[idx];
    }

    std::vector<float> getImageNormAt(int idx) const {
        uchar* img = getImageAt(idx);
        std::vector<float> v;
        for (int i = 0; i < getImgSize(); ++i)
            v.push_back((float)+img[i] / 255);
        return v;
    }

    int getLabelAt(int idx) const {
        if (idx >= num || idx < 0) throw std::runtime_error("Index out of bound!");
        return +labels[idx];
    }

    void printInfo() {
        std::cout << "Image data shape: " << num << " images * " << getImgSize() << " pixels\n";
        std::cout << "Image dimensions: (" << dimX << " * " << dimY << ") px\n";
        std::cout << "Label data shape: " << num << " labels\n\n";
    }

    void printImage(int idx) {
        if (idx >= num || idx < 0) throw std::runtime_error("Index out of bound!");
        const char color[] = " .:+#";
        uchar* img = images[idx];
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
                int pxValue = +img[y * dimX + x];
                char px = pxValue == 0
                              ? color[0]
                          : pxValue < 64
                              ? color[1]
                          : pxValue < 128
                              ? color[2]
                          : pxValue < 192
                              ? color[3]
                              : color[4];
                std::cout << px << px;
            }
            std::cout << std::endl;
        }
    }

    void printImageSml(int idx) {
        if (idx >= num || idx < 0) throw std::runtime_error("Index out of bound!");
        const char color[] = " .:+#";
        uchar* img = images[idx];
        for (int y = 0; y < dimY; y += 2) {
            for (int x = 0; x < dimX; x += 2) {
                int pxValue = (+img[y * dimX + x] +
                               +img[y * dimX + x + 1] +
                               +img[(y + 1) * dimX + x] +
                               +img[(y + 1) * dimX + x + 1]) /
                              4;
                char px = pxValue == 0
                              ? color[0]
                          : pxValue < 64
                              ? color[1]
                          : pxValue < 128
                              ? color[2]
                          : pxValue < 192
                              ? color[3]
                              : color[4];
                std::cout << px << px;
            }
            std::cout << std::endl;
        }
    }

    void printLabel(int idx) {
        int lbl = getLabelAt(idx);
        std::cout << "Label #" << idx << ": " << lbl << std::endl;
    }

    void printImgWithLabel(int idx) {
        printImage(idx);
        printLabel(idx);
    }

    ~DataSet() {
        for (int i = 0; i < num; ++i) {
            delete[] images[i];
        }
        delete[] images;
        delete[] labels;
    }
};

/**
 * @brief Get random int in the range [0, high)
 *
 * @param high Upper bound (exclusive)
 * @return int
 */
int randInt(int high) {
    int x = high;
    while (x >= high)
        x = std::rand() / ((RAND_MAX + 1u) / high);
    return x;
}

/**
 * @brief Get random float in the range [0, 1]
 *
 * @return int
 */
float randFloat() {
    return (float)std::rand() / RAND_MAX;
}

float maxOfVector(const std::vector<float>& v) {
    float max = v[0];
    for (auto& x : v)
        max = x > max ? x : max;
    return max;
}

int idxMaxOfVec(const std::vector<float>& v) {
    int idxMax = 0;
    float max = v[0];
    for (int i = 0; i < v.size(); ++i) {
        float x = v[i];
        idxMax = x > max ? i : idxMax;
        max = x > max ? x : max;
    }
    return idxMax;
}

/* Parameters Initialization */
struct LayerParams {
    std::vector<float> b;
    std::vector<std::vector<float>> w;
};

/**
 * @brief Initialize parameters in globally-defined Dimensions and Ranges
 *
 * @return std::vector<LayerParams>
 */
std::vector<LayerParams> initParams();

/**
 * @brief Randomize a layer_current-sized vector in a certain range
 *
 * @param layer
 * @return std::vector<float>
 */
std::vector<float> initParamsB(int layer);

/**
 * @brief Randomize a layer_prev * layer_current matrix in a certain range
 *
 * @param layer
 * @return std::vector<std::vector<float>>
 */
std::vector<std::vector<float>> initParamsW(int layer);

/* Predict */
std::vector<float> predict(const std::vector<float>& img, const std::vector<LayerParams>& params);

/**
 * @brief Vector addition: v1 = v1 + v2
 *
 * @param dest v1, will be updated to the new value
 * @param right v2, should have the same size as v1
 * @return std::vector<float> v1
 */
std::vector<float> vAdd(std::vector<float>& dest, const std::vector<float>& right);

std::vector<float> vMul(std::vector<float>& dest, float multiplier);

std::vector<float> vPow(std::vector<float>& dest, float exponent);

std::vector<std::vector<float>> mAdd(std::vector<std::vector<float>>& dest, const std::vector<std::vector<float>>& right);

std::vector<std::vector<float>> mMul(std::vector<std::vector<float>>& dest, float multiplier);

std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& mat);

/**
 * @brief Dot product of a vector and a matrix: new vector = v . M
 *
 * @param left v: a vector
 * @param right M: an m * n matrix, whose m should have the same size as v
 * @return std::vector<float> new vector
 */
std::vector<float> vDotM(const std::vector<float>& left, const std::vector<std::vector<float>>& right);

/**
 * @brief Outer product of two vector: new matrix = v1 x v2
 *
 * @param left v1: an m-sized vector
 * @param right v2: an n-sized vector
 * @return std::vector<std::vector<float>> new m * n matrix
 */
std::vector<std::vector<float>> vOuter(const std::vector<float>& left, const std::vector<float>& right);

/**
 * @brief Generate a d * d diagonal matrix base on a d-sized vector
 *
 * @param v
 * @return std::vector<std::vector<float>>
 */
std::vector<std::vector<float>> diagonal(const std::vector<float>& v);

/**
 * @brief Generate a d * d identity matrix
 *
 * @param dimension d
 * @return std::vector<std::vector<float>>
 */
std::vector<std::vector<float>> identity(int dimension);

/* Batch Training */
std::vector<LayerParams> trainBatch(const DataSet& trainData, int currBatchIdx, const std::vector<LayerParams>& params);
std::vector<LayerParams> gradientParams(const std::vector<float>& img, int lbl, const std::vector<LayerParams>& params);
std::vector<LayerParams> paramsAdd(std::vector<LayerParams>& dest, const std::vector<LayerParams>& right);
std::vector<LayerParams> paramsMul(std::vector<LayerParams>& dest, float multiplier);

std::vector<LayerParams> getParamsDescendOnce(const std::vector<LayerParams>& params, const std::vector<LayerParams> gradParams, float learnRate);

/* Loss (cost) function */
float sqrtLoss(const std::vector<float>& img, int lbl, const std::vector<LayerParams>& params);

float countLoss(const DataSet& data, const std::vector<LayerParams> params);
float countAccuracy(const DataSet& data, const std::vector<LayerParams> params);
