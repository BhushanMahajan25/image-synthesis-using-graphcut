# Image synthesis using Graph-cut

## Assignment description

The goal of the assignment is to synthesize an image with significantly larger dimensions from a sample image using the algorithm discussed in the paper by - Vivek Kwatra, Arno Schödl, Irfan Essa, Greg Turk, and Aaron Bobick. 2003. "Graphcut textures: image and video synthesis using graph cuts". <i>ACM Trans. Graph.</i> 22, 3 (July 2003), 277–286. DOI:https://doi.org/10.1145/882262.882264

High-level process flow is explained below:

* Create an empty image for final output. The output image size can be specified in `src/config.py`. 
* The input image is treated as a whole patch and placed randomely into the output image. Similarly, another patch which is input image is placed into the output image such that previous patch and current patch will overlap.
* Then, mincut algorithm is applied in the overlapped region. Edmonds-karp algorithm is used for maxflow which in turn gives the edges from which mincut can be achieved.
* This minimum cut generated serves as the boundary for the inclusion of pixels from the overlapped images.
* We repeat this process until we max out the dimensions of the resulting image.

## Statement of Help
### How to install dependencies : 
    `python3 -m pip install -r requirements.txt`

### How to run the code : 
    `python3 src/main.py`

The program asks user to select the image for graphcut-synthesis from menu. After the choice is given by user, prgram start synthesizing the input image.

### How to interpret the output : 
The output image will be generated inside `output` directory. The dimensions of this output image can be specified in `src/config.py`

## Implementation 
### Libraries Used
* imageio -> To perform read and write operations on images.
* cv2 -> To perform scaling operations.
* numpy -> To effectively deal with 2d and 3d matrices.
* matplotlib -> To plot graphs.
* networkx -> To create graphs from the overlapped pixels.
