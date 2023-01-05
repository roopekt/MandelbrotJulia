#pragma once

#include <opencv2/opencv.hpp>

namespace fractal
{
	namespace core
	{
		class GPUFrameBuffer
		{
		private:
			int width;
			int height;

			unsigned char* buffer;
			cv::Mat encodedBuffer;

			unsigned char* gradientColors;//device pointer
			double* gradientKeyPositions;//device pointer
			int gradientKeyCount;
			unsigned char* gradientComparer;//these three values are used to check, if different gradient is inputed and thus if it should be encoded
			int gradientSizeComparer = -1;
			double gradientStretchComparer;

			bool updateGradient(const unsigned char* const gradient, int gradientSize, double gradientStretch);//encode gradient if necessary (source values changed). returns true, if update was necessary

		public:
			GPUFrameBuffer(int width, int height);

			const cv::Mat* frame(int fractalType, double seedR, double seedI, double focusR, double focusI, double zoomWidth, unsigned int iterations, double escapeRadius, bool smooth,
				const unsigned char* const gradient, int gradientSize, double gradientStretch, const unsigned char* fractalColor);

			~GPUFrameBuffer();//calls free()
		};
	}
}