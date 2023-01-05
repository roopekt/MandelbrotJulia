#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "Core.h"
//#include "Generator.h"

using namespace cv;
using namespace std;

namespace fractal
{
    namespace core
    {
        __global__ static void kernel(unsigned char* buffer, int width, int height, int fractalType, double seedR, double seedI, double focusR, double focusI, double zoomWidth, unsigned int iterations, double escapeRadius, bool smooth,
            unsigned char* gradientColors, double* gradientKeyPositions, int gradientKeyCount, unsigned char fractalColorR, unsigned char fractalColorG, unsigned char fractalColorB)
        {
            long int pixelCount = width * height;
            int stride = blockDim.x * gridDim.x;

            double escapeRadiusSquared = escapeRadius * escapeRadius;

            double scalar = zoomWidth / width;

            double ofsetR = focusR - (width - 1) / 2 * scalar;
            double ofsetI = focusI - (height - 1) / 2 * scalar;

            for (long int pixel = (long int)blockIdx.x * blockDim.x + threadIdx.x; pixel < pixelCount; pixel += stride)
            {
                //screenspace coordinates
                int x = pixel % width;
                int y = height - 1 - (pixel - x) / width;

                //transform from screenspace to complex numbers
                double r = x * scalar + ofsetR;
                double i = y * scalar + ofsetI;

                //setup variables
                double Zr, Zi, Cr, Ci;
                if (fractalType == 0) {//!!!!!!!!!!!!!!! mandelbrotSet
                    Zr = seedR;
                    Zi = seedI;
                    Cr = r;
                    Ci = i;
                }
                else {
                    Zr = r;
                    Zi = i;
                    Cr = seedR;
                    Ci = seedI;
                }

                //iterations
                bool escapes = false;
                int I;
                for (I = 0; I < iterations; I++)
                {
                    //check if escaped
                    if (Zr * Zr + Zi * Zi >= escapeRadiusSquared) {
                        escapes = true;
                        break;
                    }

                    //square Z
                    double oldZr = Zr;
                    Zr = Zr * Zr - Zi * Zi;
                    Zi = 2 * oldZr * Zi;

                    //add C to Z
                    Zr += Cr;
                    Zi += Ci;
                }

                //calculate color
                unsigned char R;
                unsigned char G;
                unsigned char B;

                if (escapes)
                {
                    double key;
                    //if (smooth) {
                    //    //key = fmod(I - log2(log2(Zr * Zr + Zi * Zi)) + 4.0, gradientKeyPositions[gradientKeyCount - 1]);//modulus (gradient's length as base) of smoothed I
                    //    //key = I % (int)gradientKeyPositions[gradientKeyCount - 1];
                    //}
                    //else {
                    //    //key = I % (int)gradientKeyPositions[gradientKeyCount - 1];
                    //    key = fmod((double)I, (double)gradientKeyPositions[gradientKeyCount - 1]);//modulus of I
                    //}
                    //key = fmod((double)I, (double)gradientKeyPositions[gradientKeyCount - 1]);//modulus of I

                    key = I % (int)gradientKeyPositions[gradientKeyCount - 1];


                    int colorA, colorB;
                    double fraction;
                    for (int i = 1; i < gradientKeyCount; i++)
                    {
                        if (gradientKeyPositions[i] >= key) {
                            colorA = i - 1;
                            colorB = i;
                            fraction = (key - gradientKeyPositions[i - 1]) / (gradientKeyPositions[i] - gradientKeyPositions[i - 1]);
                            break;
                        }
                    }

                    R = gradientColors[colorA * 3 + 0] + (gradientColors[colorB * 3 + 0] - gradientColors[colorA * 3 + 0]) * fraction;
                    G = gradientColors[colorA * 3 + 1] + (gradientColors[colorB * 3 + 1] - gradientColors[colorA * 3 + 1]) * fraction;
                    B = gradientColors[colorA * 3 + 2] + (gradientColors[colorB * 3 + 2] - gradientColors[colorA * 3 + 2]) * fraction;
                }
                else {
                    R = fractalColorR;
                    G = fractalColorG;
                    B = fractalColorB;
                }

                //write to buffer
                buffer[3 * pixel] = B;
                buffer[3 * pixel + 1] = G;
                buffer[3 * pixel + 2] = R;
            }
        }

        bool GPUFrameBuffer::updateGradient(const unsigned char* const gradient, int gradientSize, double gradientStretch)//encode gradient if necessary (source values changed). returns true, if update was necessary
        {
            gradientSize = max(0, gradientSize);

            //check if update is necessary
            if (gradientStretch != this->gradientStretchComparer || gradientSize != this->gradientSizeComparer)
                goto Update;
            for (int i = 0; i < gradientSize; i++)
            {
                if (gradient[i] != this->gradientComparer[i])
                    goto Update;
            }
            return 0;//if all was same, don't update

        Update:
            //free old memory, except if this is first update
            if (gradientSizeComparer != -1) {
                cudaFree(gradientColors);
                cudaFree(gradientKeyPositions);
            }

            //update comparers
            gradientComparer = (unsigned char*)gradient;
            memcpy(gradientComparer, gradient, sizeof(unsigned char) * gradientSize);
            gradientSizeComparer = gradientSize;
            gradientStretchComparer = gradientStretch;

            //encode gradient (onto host). update gradientKeyCount
            gradientKeyCount = gradientSize / 4 + 1;//+1: also last color (same as first) included
            unsigned char* _gradientColors = new unsigned char[gradientKeyCount * 3];
            double* _gradientKeyPositions = new double[gradientKeyCount];
            _gradientKeyPositions[0] = 0;//first key is at 0
            for (int i = 0; i < gradientKeyCount - 1; i++)
            {
                _gradientColors[i * 3 + 0] = gradient[i * 4 + 0];
                _gradientColors[i * 3 + 1] = gradient[i * 4 + 1];
                _gradientColors[i * 3 + 2] = gradient[i * 4 + 2];

                _gradientKeyPositions[i + 1] = _gradientKeyPositions[i] + gradient[i * 4 + 3] * gradientStretch;
            }
            _gradientColors[gradientKeyCount - 3] = gradient[0];//last color is the same as first
            _gradientColors[gradientKeyCount - 2] = gradient[1];
            _gradientColors[gradientKeyCount - 1] = gradient[2];

            //copy to GPU
            cudaMalloc((void**)&gradientColors, sizeof(unsigned char) * 3 * gradientKeyCount);
            cudaMemcpy((void*)gradientColors, (void*)_gradientColors, sizeof(unsigned char) * 3 * gradientKeyCount, cudaMemcpyHostToDevice);
            cudaMalloc((void**)&gradientKeyPositions, sizeof(double) * gradientKeyCount);
            cudaMemcpy((void*)gradientKeyPositions, (void*)_gradientKeyPositions, sizeof(double) * gradientKeyCount, cudaMemcpyHostToDevice);

            //free host data
            delete[] _gradientColors;
            delete[] _gradientKeyPositions;

            return true;
        }

        GPUFrameBuffer::GPUFrameBuffer(int _width, int _height)
        {
            width = _width;
            height = _height;

            cudaMallocManaged(&buffer, sizeof(unsigned char) * (long int)width * height * 3);
            encodedBuffer = Mat(height, width, CV_8UC3, buffer);
        }

        const Mat* GPUFrameBuffer::frame(int fractalType, double seedR, double seedI, double focusR, double focusI, double zoomWidth, unsigned int iterations, double escapeRadius, bool smooth,
            const unsigned char* const gradient, int gradientSize, double gradientStretch, const unsigned char* fractalColor)
        {
            updateGradient(gradient, gradientSize, gradientStretch);

            kernel<<<512, 512>>>(buffer, width, height, fractalType, seedR, seedI, focusR, focusI, zoomWidth, iterations, escapeRadius, smooth,
                this->gradientColors, this->gradientKeyPositions, this->gradientKeyCount, fractalColor[0], fractalColor[1], fractalColor[2]);
            cudaDeviceSynchronize();

            return &encodedBuffer;
        }

        GPUFrameBuffer::~GPUFrameBuffer() {
            cudaFree(buffer);
            encodedBuffer.release();

            if (gradientSizeComparer != -1) {
                cudaFree(gradientColors);
                cudaFree(gradientKeyPositions);
            }
        }
    }
}