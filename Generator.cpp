#include "Generator.h"
#include <opencv2/opencv.hpp>
#include "Core.h"

using namespace cv;
using namespace std;
using namespace fractal;

static const double PI = 3.14159265358979323846;

FrameBuffer::FrameBuffer(int width, int height) : buffer(width, height)
{
    this->width = width;
    this->height = height;
}

const Mat* FrameBuffer::frame(int fractalType, double seedR, double seedI, double focusR, double focusI, double zoomWidth, unsigned int iterations, double escapeRadius, bool smooth,
    const unsigned char* const gradient, int gradientSize, double gradientStretch, const array<unsigned char, 3> fractalColor)
{
#ifdef _DEBUG
    if (fractalType != mandelbrotSet && fractalType != juliaSet) {
        throw invalid_argument("Invalid argument: fractalType. Expected value: 0 or 1. Received value: " + to_string(fractalType));
    }
    if (zoomWidth <= 0) {
        throw invalid_argument("Invalid argument: zoomWidth. Expected value: non-zero positive. Received value: " + to_string(zoomWidth));
    }
    if (iterations <= 0) {
        throw invalid_argument("Invalid argument: iterations. Expected value: non-zero positive. Received value: " + to_string(iterations));
    }
    if (gradientSize % 4 != 0) {
        throw invalid_argument("Invalid argument: gradientSize. Expected value: divisible by 4. Received value: " + to_string(gradientSize));
    }
    if (gradientStretch <= 0) {
        throw invalid_argument("Invalid argument: gradientStretch. Expected value: non-zero positive. Received value: " + to_string(gradientStretch));
    }
    if (escapeRadius < 0) {
        throw invalid_argument("Invalid argument: escapeRadius. Expected value: positive. Received value: " + to_string(escapeRadius));
    }
#endif

    return buffer.frame(fractalType, seedR, seedI, focusR, focusI, zoomWidth, iterations, escapeRadius, smooth, gradient, gradientSize, gradientStretch, &fractalColor[0]);

    //todo:
    //
    //smooth
}

Video::Video(int width, int height, double FPS, string savePath, int fourcc, int fractalType, double seedR, double seedI, double focusR, double focusI, double zoomWidth, unsigned int iterations,
    const unsigned char* const gradient, int gradientSize, double gradientStretch, const array<unsigned char, 3> fractalColor, double escapeRadius, bool smooth) : Video(width, height, FPS, savePath, fourcc)//sets all parameters, and calls other constructor
{
#ifdef _DEBUG
    if (width <= 0) {
        throw invalid_argument("Invalid argument: width. Expected value: non-zero positive. Received value: " + to_string(width));
    }
    if (height <= 0) {
        throw invalid_argument("Invalid argument: height. Expected value: non-zero positive. Received value: " + to_string(height));
    }
    if (FPS <= 0) {
        throw invalid_argument("Invalid argument: FPS. Expected value: non-zero positive. Received value: " + to_string(FPS));
    }
    if (fractalType != mandelbrotSet && fractalType != juliaSet) {
        throw invalid_argument("Invalid argument: fractalType. Expected value: 0 or 1. Received value: " + to_string(fractalType));
    }
    if (zoomWidth <= 0) {
        throw invalid_argument("Invalid argument: zoomWidth. Expected value: non-zero positive. Received value: " + to_string(zoomWidth));
    }
    if (iterations <= 0) {
        throw invalid_argument("Invalid argument: iterations. Expected value: non-zero positive. Received value: " + to_string(iterations));
    }
    if (gradientSize % 4 != 0) {
        throw invalid_argument("Invalid argument: gradientSize. Expected value: divisible by 4. Received value: " + to_string(gradientSize));
    }
    if (gradientStretch <= 0) {
        throw invalid_argument("Invalid argument: gradientStretch. Expected value: non-zero positive. Received value: " + to_string(gradientStretch));
    }
    if (escapeRadius < 0) {
        throw invalid_argument("Invalid argument: escapeRadius. Expected value: positive. Received value: " + to_string(escapeRadius));
    }
#endif

    this->fractalType = fractalType;
    this->seedR = seedR;
    this->seedI = seedI;
    this->focusR = focusR;
    this->focusI = focusI;
    this->zoomWidth = zoomWidth;
    this->iterations = iterations;

    this->gradient = new unsigned char[gradientSize];
    memcpy((void*)this->gradient, (void*)gradient, sizeof(unsigned char) * gradientSize);

    this->gradientSize = max(0, gradientSize);
    this->gradientStretch = gradientStretch;
    this->fractalColor = fractalColor;
    this->escapeRadius = escapeRadius;
    this->smooth = smooth;
}

Video::Video(int width, int height, double FPS, string savePath, int fourcc) : frameBuffer(width, height), videoWriter(savePath, fourcc, FPS, Size(width, height)) {//constructs frameBuffer and videoWriter
#ifdef _DEBUG
    if (width <= 0) {
        throw invalid_argument("Invalid argument: width. Expected value: non-zero positive. Received value: " + to_string(width));
    }
    if (height <= 0) {
        throw invalid_argument("Invalid argument: height. Expected value: non-zero positive. Received value: " + to_string(height));
    }
    if (FPS <= 0) {
        throw invalid_argument("Invalid argument: FPS. Expected value: non-zero positive. Received value: " + to_string(FPS));
    }
#endif

    this->FPS = FPS;
    if (gradientSize == -1) {
        this->gradient = new unsigned char[8]{ 255, 0, 0, 1, 255, 255, 255, 1 };
        this->gradientSize = 8;
    }
}

Video::~Video() {
    videoWriter.release();
    delete[] gradient;
}

const Mat* Video::newFrame()
{
    const Mat* frame = frameBuffer.frame(fractalType, seedR, seedI, focusR, focusI, zoomWidth, iterations, escapeRadius, smooth, gradient, gradientSize, gradientStretch, fractalColor);
    videoWriter.write(*frame);

    //execute updateRule
    double oldSpiralR;
    switch (updateRule)
    {
    case updateRule_none:
        break;

    case updateRule_zoom:
        zoomWidth *= zoomMultiplyer;
        break;

    case updateRule_spiralZoom:
        zoomWidth *= zoomMultiplyer;
        oldSpiralR = spiralR;
        spiralR = spiralR * spiralDeltaR - spiralI * spiralDeltaI;
        spiralI = oldSpiralR * spiralDeltaI + spiralI * spiralDeltaR;
        seedR = spiralCentreR + spiralR;
        seedI = spiralCentreI + spiralI;
        break;

    default:
        throw "Unknown updateRule: " + to_string(updateRule);
    }

    return frame;
}

void Video::release() {
    videoWriter.release();
}

void Video::setUpdateRule_zoom(double zoomSpeed) {
#ifdef _DEBUG
    if (zoomSpeed <= 0) {
        throw invalid_argument("Invalid argument: zoomSpeed. Expected value: non-zero positive. Received value: " + to_string(zoomSpeed));
    }
#endif
    updateRule = updateRule_zoom;

    zoomMultiplyer = pow(1.0 / zoomSpeed, 1.0 / FPS);
}

void Video::setUpdateRule_spiralZoom(double zoomSpeed, double spiralCentreR, double spiralCentreI, double spiralStartingRadius, double spiralRotatingSpeed, double spiralStartingAngle) {
#ifdef _DEBUG
    if (zoomSpeed <= 0) {
        throw invalid_argument("Invalid argument: zoomSpeed. Expected value: non-zero positive. Received value: " + to_string(zoomSpeed));
    }
    if (spiralStartingRadius < 0) {
        throw invalid_argument("Invalid argument: spiralStartingRadius. Expected value: positive. Received value: " + to_string(spiralStartingRadius));
    }
#endif
    updateRule = updateRule_spiralZoom;

    zoomMultiplyer = pow(1.0 / zoomSpeed, 1.0 / FPS);
    this->spiralCentreR = spiralCentreR;
    this->spiralCentreI = spiralCentreI;

    const double degree2radian = 1.0 / 360.0 * 2.0 * PI;

    spiralDeltaR = zoomMultiplyer * zoomMultiplyer * cos(spiralRotatingSpeed * degree2radian / FPS);
    spiralDeltaI = zoomMultiplyer * zoomMultiplyer * sin(spiralRotatingSpeed * degree2radian / FPS);

    spiralR = spiralStartingRadius * cos(spiralStartingAngle * degree2radian);
    spiralI = spiralStartingRadius * sin(spiralStartingAngle * degree2radian);

    seedR = spiralCentreR + spiralR;
    seedI = spiralCentreI + spiralI;
}