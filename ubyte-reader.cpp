#include "ubyte-reader.h"

// Read ubyte
// code from https://stackoverflow.com/a/33384846

auto reverseInt = [](int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
};

uchar **readMnistImages(std::string fullPath, int &numOfImages, int &numOfRows, int &numOfCols) {
    std::ifstream file(fullPath, std::ios::binary);

    if (file.is_open()) {
        int magicNumber = 0;

        file.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);

        if (magicNumber != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char *)&numOfImages, sizeof(numOfImages)), numOfImages = reverseInt(numOfImages);
        file.read((char *)&numOfRows, sizeof(numOfRows)), numOfRows = reverseInt(numOfRows);
        file.read((char *)&numOfCols, sizeof(numOfCols)), numOfCols = reverseInt(numOfCols);

        int sizeOfImage = numOfRows * numOfCols;

        uchar **_dataset = new uchar *[numOfImages];
        for (int i = 0; i < numOfImages; i++) {
            _dataset[i] = new uchar[sizeOfImage];
            file.read((char *)_dataset[i], sizeOfImage);
        }
        return _dataset;
    } else {
        throw std::runtime_error("Cannot open file `" + fullPath + "`!");
    }
}

uchar *readMnistLabels(std::string fullPath, int &numOfLabels) {
    std::ifstream file(fullPath, std::ios::binary);

    if (file.is_open()) {
        int magicNumber = 0;
        file.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);

        if (magicNumber != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *)&numOfLabels, sizeof(numOfLabels)), numOfLabels = reverseInt(numOfLabels);

        uchar *_dataset = new uchar[numOfLabels];
        for (int i = 0; i < numOfLabels; i++) {
            file.read((char *)&_dataset[i], 1);
        }
        return _dataset;
    } else {
        throw std::runtime_error("Unable to open file `" + fullPath + "`!");
    }
}
// end of borrowed code