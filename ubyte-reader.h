#include <fstream>
#include <string>

typedef unsigned char uchar;
uchar **readMnistImages(std::string fullPath, int &numOfImages, int &numOfRows, int &numOfCols);
uchar *readMnistLabels(std::string fullPath, int &numOfLabels);
