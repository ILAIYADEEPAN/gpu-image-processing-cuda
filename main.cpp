#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    if (argc < 2) {
        cout << "Usage: ./image_processor <image_path>" << endl;
        return -1;
    }

    string imagePath = argv[1];

    // Load image
    Mat inputImage = imread(imagePath);

    if (inputImage.empty()) {
        cout << "Error loading image!" << endl;
        return -1;
    }

    cout << "Image loaded successfully!" << endl;

    // Upload image to GPU
    cuda::GpuMat gpuInput;
    gpuInput.upload(inputImage);

    // -----------------------------
    // GPU Grayscale Conversion
    // -----------------------------
    cuda::GpuMat gpuGray;

    double start = (double)getTickCount();

    cuda::cvtColor(gpuInput, gpuGray, COLOR_BGR2GRAY);

    double gpuTimeGray =
        ((double)getTickCount() - start) / getTickFrequency();

    // Download result
    Mat grayImage;
    gpuGray.download(grayImage);

    imwrite("grayscale_output.jpg", grayImage);

    // -----------------------------
    // GPU Gaussian Blur
    // -----------------------------
    cuda::GpuMat gpuBlur;

    start = (double)getTickCount();

    Ptr<cuda::Filter> gaussianFilter =
        cuda::createGaussianFilter(
            gpuGray.type(),
            gpuBlur.type(),
            Size(15, 15),
            0
        );

    gaussianFilter->apply(gpuGray, gpuBlur);

    double gpuTimeBlur =
        ((double)getTickCount() - start) / getTickFrequency();

    Mat blurImage;
    gpuBlur.download(blurImage);

    imwrite("blur_output.jpg", blurImage);

    // -----------------------------
    // GPU Edge Detection
    // -----------------------------
    cuda::GpuMat gpuEdges;

    start = (double)getTickCount();

    Ptr<cuda::CannyEdgeDetector> canny =
        cuda::createCannyEdgeDetector(50.0, 150.0);

    canny->detect(gpuGray, gpuEdges);

    double gpuTimeEdge =
        ((double)getTickCount() - start) / getTickFrequency();

    Mat edgeImage;
    gpuEdges.download(edgeImage);

    imwrite("edge_output.jpg", edgeImage);

    // -----------------------------
    // Display Results
    // -----------------------------
    cout << "\nGPU Processing Completed!" << endl;

    cout << "Grayscale Time: "
         << gpuTimeGray * 1000
         << " ms" << endl;

    cout << "Gaussian Blur Time: "
         << gpuTimeBlur * 1000
         << " ms" << endl;

    cout << "Edge Detection Time: "
         << gpuTimeEdge * 1000
         << " ms" << endl;

    cout << "\nOutput images saved successfully!" << endl;

    return 0;
}
