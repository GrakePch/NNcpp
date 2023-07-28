#include "main.h"
/*
    Overall logic inspired by
    https://space.bilibili.com/28496477/channel/seriesdetail?sid=1267418
*/

std::string DATASET_PATH = "./MNIST/";
std::string TRAIN_IMG_PATH = DATASET_PATH + "train-images.idx3-ubyte";
std::string TRAIN_LBL_PATH = DATASET_PATH + "train-labels.idx1-ubyte";
std::string TEST_IMG_PATH = DATASET_PATH + "t10k-images.idx3-ubyte";
std::string TEST_LBL_PATH = DATASET_PATH + "t10k-labels.idx1-ubyte";
int VALID_NUM = 10000;
int LAYER_NUM = 3;
int DIMENSIONS[] = {28 * 28, 100, 10};
fn* ACTIVATIONS[] = {bypass, tanh, softmax};
dfn* DIFFERENCIALS[] = {dBypass, dTanh, dSoftmax};
float LEARN_RATE = std::pow(10, -1);
int BATCH_SIZE = 100;
int EPOCH_NUM = 5;
std::vector<std::vector<float>> MAT_ONE_HOT;

// Define ranges for parameter initialization
Range DISTRIBUTION[] = {
    {},  // keep empty
    {
        {0, 0},  // b
        {-(float)std::sqrt(6.0 / (DIMENSIONS[0] + DIMENSIONS[1])),
         (float)std::sqrt(6.0 / (DIMENSIONS[0] + DIMENSIONS[1]))},  // w
    },
    {
        {0, 0},  // b
        {-(float)std::sqrt(6.0 / (DIMENSIONS[1] + DIMENSIONS[2])),
         (float)std::sqrt(6.0 / (DIMENSIONS[1] + DIMENSIONS[2]))},  // w
    },
};

int main() {
    std::srand(std::time(nullptr));

    /* Import MNIST data */
    DataSet trainData, validData, testData;

    int trainAndValidNum;
    uchar** trainAndValidImgs = readMnistImages(TRAIN_IMG_PATH, trainAndValidNum, trainData.dimX, trainData.dimY);
    uchar* trainAndValidLbls = readMnistLabels(TRAIN_LBL_PATH, trainAndValidNum);
    trainData.num = trainAndValidNum - VALID_NUM;
    validData.num = VALID_NUM;
    validData.dimX = trainData.dimX;
    validData.dimY = trainData.dimY;

    trainData.images = new uchar*[trainData.num];
    trainData.labels = new uchar[trainData.num];
    validData.images = new uchar*[validData.num];
    validData.labels = new uchar[validData.num];
    for (int i = 0; i < trainData.num; ++i) {
        trainData.images[i] = trainAndValidImgs[i];
        trainData.labels[i] = trainAndValidLbls[i];
    }
    for (int i = 0; i < validData.num; ++i) {
        validData.images[i] = trainAndValidImgs[i + trainData.num];
        validData.labels[i] = trainAndValidLbls[i + trainData.num];
    }
    delete[] trainAndValidImgs;
    delete[] trainAndValidLbls;

    testData.images = readMnistImages(TEST_IMG_PATH, testData.num, testData.dimX, testData.dimY);
    testData.labels = readMnistLabels(TEST_LBL_PATH, testData.num);

    std::cout << "===TRAINING DATA INFO===\n";
    trainData.printInfo();
    std::cout << "===VALIDATION DATA INFO===\n";
    validData.printInfo();
    std::cout << "===TEST DATA INFO===\n";
    testData.printInfo();

    /* Initialize parameters */
    DIMENSIONS[0] = trainData.getImgSize();
    auto params = initParams();

    int currEpoch = 0;
    std::vector<float>
        listTrainLoss,
        listValidLoss,
        listTrainAccuracy,
        listValidAccuracy;
    /* Initialize one-hot matrix */
    MAT_ONE_HOT = identity(DIMENSIONS[LAYER_NUM - 1]);

    /* Training */
    int batchNum = trainData.num / BATCH_SIZE;
    for (int epoch = 0; epoch < EPOCH_NUM; ++epoch) {
        printf("Start Epoch %d/%d\n", epoch + 1, EPOCH_NUM);
        for (int i = 0; i < batchNum; ++i) {
            if (i % 100 == 99)
                printf("Running Batch %d/%d\n", i + 1, batchNum);
            auto gradientAvg = trainBatch(trainData, i, params);
            params = getParamsDescendOnce(params, gradientAvg, LEARN_RATE);
        }
        currEpoch++;
        listTrainLoss.push_back(countLoss(trainData, params));
        listValidLoss.push_back(countLoss(validData, params));
        listTrainAccuracy.push_back(countAccuracy(trainData, params));
        listValidAccuracy.push_back(countAccuracy(validData, params));
    }
    printf("Finished\n");
    std::cout << "Training Loss:\n";
    for (auto& e : listTrainLoss) std::cout << e << " ";
    std::cout << "\nValidation Loss:\n";
    for (auto& e : listValidLoss) std::cout << e << " ";
    std::cout << "\nTraining Accuracy:\n";
    for (auto& e : listTrainAccuracy) std::cout << e << " ";
    std::cout << "\nValidation Accuracy:\n";
    for (auto& e : listValidAccuracy) std::cout << e << " ";
    std::cout << std::endl;

    int testRunNum = 4;
    printf("===Run %d Prediction on test data===\n", testRunNum);
    for (int i = 0; i < testRunNum; ++i) {
        int idx = randInt(testData.num);
        std::vector<float> result = predict(testData.getImageNormAt(idx), params);
        int predictAns = idxMaxOfVec(result);
        testData.printImage(idx);
        printf("Predict: %d (%f%) | Label: %d\n", predictAns, result[predictAns] * 100, testData.getLabelAt(idx));
    }
}

std::vector<LayerParams> initParams() {
    std::vector<LayerParams> params;
    for (int i = 0; i < LAYER_NUM; ++i) {
        LayerParams currLayerParams;
        currLayerParams.b = initParamsB(i);
        currLayerParams.w = initParamsW(i);
        params.push_back(currLayerParams);
    }
    return params;
}

std::vector<float> initParamsB(int layer) {
    std::vector<float> v;
    auto range = DISTRIBUTION[layer].b;
    for (int i = 0; i < DIMENSIONS[layer]; ++i)
        v.push_back(randFloat() * (range[1] - range[0]) + range[0]);
    return v;
}

std::vector<std::vector<float>> initParamsW(int layer) {
    // layer should > 0
    std::vector<std::vector<float>> m;
    auto range = DISTRIBUTION[layer].w;
    for (int i = 0; i < DIMENSIONS[layer - 1]; ++i) {
        std::vector<float> v;
        for (int j = 0; j < DIMENSIONS[layer]; ++j) {
            v.push_back(randFloat() * (range[1] - range[0]) + range[0]);
        }
        m.push_back(v);
    }
    return m;
}

std::vector<float> predict(const std::vector<float>& img, const std::vector<LayerParams>& params) {
    std::vector<float> l_in(img);
    std::vector<float> l_out = ACTIVATIONS[0](l_in);
    for (int layer = 1; layer < LAYER_NUM; ++layer) {
        l_in = vDotM(l_out, params[layer].w);
        vAdd(l_in, params[layer].b);
        l_out = ACTIVATIONS[layer](l_in);
    }
    return l_out;
}

std::vector<float> bypass(const std::vector<float>& v) { return v; }

std::vector<float> tanh(const std::vector<float>& v) {
    std::vector<float> newV;
    for (auto& x : v)
        newV.push_back(std::tanh(x));
    return newV;
}

std::vector<float> softmax(const std::vector<float>& v) {
    std::vector<float> newV;
    float max = maxOfVector(v);
    float expSum = 0;
    for (auto& x : v) {
        float temp = std::exp(x - max);
        expSum += temp;
        newV.push_back(temp);
    }
    for (auto& x : newV) {
        x /= expSum;
    }
    return newV;
}

std::vector<std::vector<float>> dBypass(const std::vector<float>& v) {
    return identity(v.size());
};

std::vector<std::vector<float>> dTanh(const std::vector<float>& v) {
    std::vector<float> newV(v);
    for (int i = 0; i < newV.size(); ++i) {
        newV[i] = 1 / pow(std::cosh(newV[i]), 2);  // 1 / cosh(x)^2
    }
    return diagonal(newV);
}

std::vector<std::vector<float>> dSoftmax(const std::vector<float>& v) {
    auto sm = softmax(v);
    auto diagSm = diagonal(sm);
    auto outerSm = vOuter(sm, sm);
    return mAdd(diagSm, mMul(outerSm, -1));  // diag(sm) - sm x sm
}

std::vector<float> vAdd(std::vector<float>& dest, const std::vector<float>& right) {
    if (dest.size() != right.size())
        throw std::runtime_error("Two vectors should have the same size!");
    for (int i = 0; i < dest.size(); ++i) {
        dest[i] += right[i];
    }
    return dest;
}

std::vector<float> vMul(std::vector<float>& dest, float multiplier) {
    for (int i = 0; i < dest.size(); ++i) {
        dest[i] *= multiplier;
    }
    return dest;
}

std::vector<float> vPow(std::vector<float>& dest, float exponent) {
    for (int i = 0; i < dest.size(); ++i) {
        dest[i] = std::pow(dest[i], exponent);
    }
    return dest;
}

std::vector<std::vector<float>> mAdd(std::vector<std::vector<float>>& dest, const std::vector<std::vector<float>>& right) {
    if (dest.size() != right.size())
        throw std::runtime_error("Two matrices should have the same dimension!");
    for (int i = 0; i < dest.size(); ++i) {
        vAdd(dest[i], right[i]);
    }
    return dest;
}

std::vector<std::vector<float>> mMul(std::vector<std::vector<float>>& dest, float multiplier) {
    for (int i = 0; i < dest.size(); ++i) {
        vMul(dest[i], multiplier);
    }
    return dest;
}

std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& mat) {
    std::vector<std::vector<float>> matT;
    for (int c = 0; c < mat[0].size(); ++c) {
        std::vector<float> newRow;
        for (int r = 0; r < mat.size(); ++r) {
            newRow.push_back(mat[r][c]);
        }
        matT.push_back(newRow);
    }
    return matT;
}

std::vector<float> vDotM(const std::vector<float>& left, const std::vector<std::vector<float>>& right) {
    if (left.size() != right.size()) {
        printf("Vector size: %d | Matrix Dim: %d*%d\n", left.size(), right.size(), right[0].size());
        throw std::runtime_error("Vector and columns of matrix should have the same size!");
    }
    std::vector<float> result;
    for (int c = 0; c < right[0].size(); ++c) {
        float temp = 0;
        for (int r = 0; r < right.size(); ++r) {
            temp += left[r] * right[r][c];
        }
        result.push_back(temp);
    }
    return result;
}

std::vector<std::vector<float>> vOuter(const std::vector<float>& left, const std::vector<float>& right) {
    std::vector<std::vector<float>> result;
    for (int r = 0; r < left.size(); ++r) {
        std::vector<float> rightCopied(right);
        vMul(rightCopied, left[r]);
        result.push_back(rightCopied);
    }
    return result;
}

std::vector<std::vector<float>> diagonal(const std::vector<float>& v) {
    std::vector<std::vector<float>> result;
    int dimension = v.size();
    for (int r = 0; r < dimension; ++r) {
        std::vector<float> row;
        for (int c = 0; c < dimension; ++c) {
            row.push_back(r == c ? v[r] : 0);
        }
        result.push_back(row);
    }
    return result;
}

std::vector<std::vector<float>> identity(int dimension) {
    std::vector<std::vector<float>> result;
    for (int r = 0; r < dimension; ++r) {
        std::vector<float> row;
        for (int c = 0; c < dimension; ++c) {
            row.push_back(r == c ? 1 : 0);
        }
        result.push_back(row);
    }
    return result;
}

std::vector<LayerParams> trainBatch(const DataSet& trainData, int currBatchIdx, const std::vector<LayerParams>& params) {
    int batHeadIdx = currBatchIdx * BATCH_SIZE;
    auto gradientAccu = gradientParams(
        trainData.getImageNormAt(batHeadIdx),
        trainData.getLabelAt(batHeadIdx),
        params);
    for (int i = 1; i < BATCH_SIZE; ++i) {
        auto gradientTemp = gradientParams(
            trainData.getImageNormAt(batHeadIdx + i),
            trainData.getLabelAt(batHeadIdx + i),
            params);
        paramsAdd(gradientAccu, gradientTemp);
    }
    paramsMul(gradientAccu, 1.0 / BATCH_SIZE);  // Get average gradient of params
    return gradientAccu;
}

std::vector<LayerParams> gradientParams(const std::vector<float>& img, int lbl, const std::vector<LayerParams>& params) {
    // Predict forward
    std::vector<std::vector<float>> listL_in = {std::vector<float>(img)};
    std::vector<std::vector<float>> listL_out = {ACTIVATIONS[0](listL_in[0])};
    for (int layer = 1; layer < LAYER_NUM; ++layer) {
        auto l_in = vDotM(listL_out[layer - 1], params[layer].w);
        vAdd(l_in, params[layer].b);
        auto l_out = ACTIVATIONS[layer](l_in);
        listL_in.push_back(l_in);
        listL_out.push_back(l_out);
    }

    // C    = (L[l] - y)^2
    // L[l] = A(z[l])
    // z[l] = w[l]L[l-1] + b[l]
    // C   : cost
    // L[l]: out value of layer l
    // A   : activation function

    //   dC       dz[l]     dL[l]      dC
    // ------- = ------- * ------- * -------
    //  dw[l]     dw[l]     dz[l]     dL[l]

    //   dC       dz[l]     dL[l]      dC
    // ------- = ------- * ------- * -------
    //  db[l]     db[l]     dz[l]     dL[l]

    //              dC
    // d_layer := ------- = 2(L[l] - y) = 2 * L[l] + (-2) * y
    //             dL[l]
    std::vector<float> dLayer(listL_out[listL_out.size() - 1]);
    vMul(dLayer, 2);
    std::vector<float> vecOntHot(MAT_ONE_HOT[lbl]);
    vMul(vecOntHot, -2);
    vAdd(dLayer, vecOntHot);

    std::vector<LayerParams> gradientResult;
    for (int i = 0; i < LAYER_NUM; i++)
        gradientResult.push_back({});

    // Back propagation
    for (int layer = LAYER_NUM - 1; layer > 0; --layer) {
        //  dL[l]
        // ------- = A'(z[l])
        //  dz[l]

        //             dL[l]      dC
        // d_layer := ------- * ------- = A'(z[l]) * 2(L[l] - y) = A'(z[l]) * d_layer
        //             dz[l]     dL[l]
        dLayer = vDotM(dLayer, transpose(DIFFERENCIALS[layer](listL_in[layer])));
        //  dz[l]
        // ------- = 1
        //  db[l]

        //          dC       dL[l]      dC
        // 'b' := ------- = ------- * ------- = d_layer
        //         db[l]     dz[l]     dL[l]
        gradientResult[layer].b = std::vector<float>(dLayer);
        //  dz[l]
        // ------- = L[l - 1]
        //  dw[l]

        //          dC                  dL[l]      dC
        // 'w' := ------- = L[l - 1] * ------- * ------- = L[l - 1] * d_layer
        //         dw[l]                dz[l]     dL[l]
        gradientResult[layer].w = vOuter(listL_out[layer - 1], dLayer);

        // Update d_layer for next iteration (previous layer)
        //  dz[l]
        // ------- = w[l]
        // dL[l-1]

        //              dC              dL[l]      dC
        // d_layer := ------- = w[l] * ------- * ------- = w[l] * d_layer
        //            dL[l-1]           dz[l]     dL[l]
        dLayer = vDotM(dLayer, transpose(params[layer].w));
    }

    return gradientResult;
}

std::vector<LayerParams> paramsAdd(std::vector<LayerParams>& dest, const std::vector<LayerParams>& right) {
    for (int l = 1; l < dest.size(); ++l) {
        vAdd(dest[l].b, right[l].b);
        mAdd(dest[l].w, right[l].w);
    }
    return dest;
}

std::vector<LayerParams> paramsMul(std::vector<LayerParams>& dest, float multiplier) {
    for (int l = 1; l < dest.size(); ++l) {
        vMul(dest[l].b, multiplier);
        mMul(dest[l].w, multiplier);
    }
    return dest;
}

float sqrtLoss(const std::vector<float>& img, int lbl, const std::vector<LayerParams>& params) {
    // Loss = sigma( (prediction - y)^2 )
    // y is the ideal value
    auto yPred = predict(img, params);
    auto y = MAT_ONE_HOT[lbl];
    float sumOfSqrtErr = 0;
    for (int i = 0; i < yPred.size(); ++i) {
        float err = y[i] - yPred[i];
        sumOfSqrtErr += err * err;
    }
    return sumOfSqrtErr;
}

std::vector<LayerParams> getParamsDescendOnce(const std::vector<LayerParams>& params, const std::vector<LayerParams> gradParams, float learnRate) {
    // get new Params := p_old - rate * p_gradient
    std::vector<LayerParams> newParams(params);
    std::vector<LayerParams> gradParamCopied(gradParams);
    paramsAdd(newParams, paramsMul(gradParamCopied, -1 * learnRate));
    return newParams;
}

float countLoss(const DataSet& data, const std::vector<LayerParams> params) {
    float lossAccu = 0;
    for (int i = 0; i < data.num; ++i)
        lossAccu += sqrtLoss(data.getImageNormAt(i), data.getLabelAt(i), params);
    return lossAccu / data.num * 10000;
}

float countAccuracy(const DataSet& data, const std::vector<LayerParams> params) {
    int correctCounter = 0;
    for (int i = 0; i < data.num; ++i)
        correctCounter += idxMaxOfVec(predict(data.getImageNormAt(i), params)) == data.getLabelAt(i) ? 1 : 0;
    return (float)correctCounter / data.num;
}
