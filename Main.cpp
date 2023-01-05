#include <opencv2/opencv.hpp>
#include "Generator.h"
#include <array>

#include "Core.h"

using namespace cv;
using namespace fractal;
using namespace std;

#pragma region parameters
const int width = 512;
const int height = width * .6;
const bool preview = 1;
const bool saveImage = 0;
const string savePath = "D:/Roope/KuvatJaVideot/debugVideo.mp4";

//video only
const bool renderVideo = true;//true: render video, false: render image
const float FPS = 30;
const int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
const double videoLength = 40;
const double zoomSpeed = 1.5;
const double spiralCentreR = -1.2522374288795870649236097248840843509256017527419668;
const double spiralCentreI = 0.3451566295671664004327107145217337363647407791740870;
const double spiralStartingRadius = .4 * 10;
const double spiralRotatingSpeed = 200;
const double spiralStartingAngle = 0;


const int fractalType = juliaSet;
const float seedR = 0, seedI = 0;

const double focusR = 0, focusI = 0;
const double zoomWidth = 10;
const int iterations = 256;

const int gradientSize = 16;
const unsigned char* const gradient = new unsigned char[gradientSize] {
    0, 14, 105, 2,//R, G, B, weight
    130, 203, 255, 1,
    227, 223, 20, 2,
    204, 149, 20, 1
};
const double gradientStretch = 5;
const array<unsigned char, 3> fractalColor = { 0, 0, 0 };

const double escapeRadius = 4;
const bool smooth = 0;
#pragma endregion

int main()
{
    try
    {
        if (renderVideo)
        {
            fractal::Video video(width, height, FPS, savePath, fourcc, fractalType, seedR, seedI, focusR, focusI, zoomWidth, iterations, gradient, gradientSize, gradientStretch, fractalColor, escapeRadius, smooth);
            delete[] gradient;
            video.setUpdateRule_spiralZoom(zoomSpeed, spiralCentreR, spiralCentreI, spiralStartingRadius, spiralRotatingSpeed, spiralStartingAngle);
            for (int i = 0; i < (int)(FPS * videoLength); i++)
            {
                //render and show frame
                if (preview) {
                    imshow("preview", *video.newFrame());
                    waitKey(1);
                }
                else {
                    video.newFrame();
                }

                //log progress
                cout << i + 1 << " / " << (int)(FPS * videoLength) << " frames done\r";
                cout.flush();
            }

            video.release();
            cout << endl;
        }
        else
        {
            FrameBuffer frameBuffer(width, height);
            const Mat* image = frameBuffer.frame(fractalType, seedR, seedI, focusR, focusI, zoomWidth, iterations, escapeRadius, smooth, gradient, gradientSize, gradientStretch, fractalColor);
            delete[] gradient;

            if (saveImage) {
                imwrite(savePath, *image);
            }

            if (preview) {
                imshow("preview", *image);
                waitKey(0);
            }
        }

        cout << "Done\n";
    }
    catch (const exception& exception) {
        cerr << "error:\n" << exception.what() << endl;
        return -1;
    }

    cin.get();
}