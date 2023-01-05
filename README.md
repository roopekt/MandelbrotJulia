# MandelbrotJulia
A video renderer for mandelbrot and julia fractals. The frames are rendered on the GPU using CUDA (weird choice, I know).

![an animated julia set](https://github.com/roopekt/MandelbrotJulia/blob/ReadmeData/ReadmeData/juliaGIF.gif)

The above animated Julia set has been produced by having the seed of the fractal (the constant term) spiral down to a point, and zooming into said point at the same speed.

## How it works 

Frames are rendered one after another, independently of each other. Color of each pixel is calculated on the GPU in parallel:

1. The pixel figures out where it is on the complex plane (with some scaling and translation)
2. $Z$ and $C$ are initialized to some complex values (one to a frame specific seed, one to the pixel's location on the complex plane, depending on the type of fractal)
3. The equation $Z = Z^2 + C$ is iterated until max iterations is reached or magnitude of $Z$ exceeds 2 (we know $Z$ will evetually explode to infinity)
4. Based on the number of iterations it took to confirm explosion, a color is sampled from a repeating gradient. If max iterations was reached, the color will be black
5. the color is written to a frame buffer

Finally, OpenCV is used to write the frame to a file.

## License 

This project is distributed under the MIT License. See `LICENSE.txt` for more information.

![a picture of a mandebrot set](https://github.com/roopekt/MandelbrotJulia/blob/ReadmeData/ReadmeData/mandelbrot.png)
