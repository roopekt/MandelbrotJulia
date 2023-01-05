#pragma once

#include <opencv2/opencv.hpp>
#include "Core.h"
#include <array>

namespace fractal
{
    const int mandelbrotSet = 0;
    const int juliaSet = 1;

    class FrameBuffer {
    private:
        int width;
        int height;
        fractal::core::GPUFrameBuffer buffer;

    public:
        FrameBuffer(int width, int height);

        const cv::Mat* frame(int fractalType, double seedR, double seedI, double focusR, double focusI, double zoomWidth, unsigned int iterations, double escapeRadius, bool smooth,
            const unsigned char* const gradient, int gradientSize, double gradientStretch, const std::array<unsigned char, 3> fractalColor);
    };

    class Video
    {
    private:
        FrameBuffer frameBuffer;
        cv::VideoWriter videoWriter;

        double FPS;

        //updaterule parameters and variables
        int updateRule = updateRule_none;
        double zoomMultiplyer = 1;//per frame
        double spiralCentreR = 0, spiralCentreI = 0;
        double spiralR = 0, spiralI = 0;//multiply this by spiralDelta every frame
        double spiralDeltaR = 0, spiralDeltaI = 0;

        //updaterule types
        static const int updateRule_none = 0;
        static const int updateRule_zoom = 1;
        static const int updateRule_spiralZoom = 2;

    public:
        int fractalType = 0;
        double seedR = 0, seedI = 0;
        double focusR = 0, focusI = 0;
        double zoomWidth = 5;
        int iterations = 32;
        unsigned char* gradient;
        int gradientSize = -1;
        double gradientStretch = 5;
        std::array<unsigned char, 3> fractalColor = { 0, 0, 0 };
        double escapeRadius = 4;
        bool smooth = true;

        Video(int width, int height, double FPS, std::string savePath, int fourcc, int fractalType, double seedR, double seedI, double focusR, double focusI, double zoomWidth, unsigned int iterations,
            const unsigned char* const gradient, int gradientSize, double gradientStretch, const std::array<unsigned char, 3> fractalColor, double escapeRadius, bool smooth);//sets all parameters, and calls other constructor

        Video(int width, int height, double FPS, std::string savePath, int fourcc);//constructs frameBuffer and videoWriter

        ~Video();//calls release

        const cv::Mat* newFrame();

        void release();

        void setUpdateRule_zoom(double zoomSpeed);

        void setUpdateRule_spiralZoom(double zoomSpeed, double spiralCentreR, double spiralCentreI, double spiralStartingRadius, double spiralRotatingSpeed, double spiralStartingAngle = 0);
    };
}