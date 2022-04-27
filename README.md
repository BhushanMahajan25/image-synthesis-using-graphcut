# Image synthesis using Graph-cut

## Assignment description

The goal of the assignment is to synthesize an image with significantly larger dimensions from a sample image using the algorithm discussed in the paper by - Vivek Kwatra, Arno Schödl, Irfan Essa, Greg Turk, and Aaron Bobick. 2003. "Graphcut textures: image and video synthesis using graph cuts". <i>ACM Trans. Graph.</i> 22, 3 (July 2003), 277–286. DOI:https://doi.org/10.1145/882262.882264

The patches were placed using the entire patch matching strategy. Using this strategy, the patches are first matched with the pre-existing patches and the best fit patch is picked. This patch picking works on probability function.

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

## Project output
1. input: `strawberries.gif` -> output: `out-strawberries.gif`

![input-1](https://github.com/BhushanMahajan25/image-synthesis-using-graphcut/blob/main/input/strawberries.gif?raw=true) | ![output-1](https://github.com/BhushanMahajan25/image-synthesis-using-graphcut/blob/main/output/out-strawberries.jpeg?raw=true)

2. input: `akeyboard_small.gif` -> output: `out-akeyboard_small.gif`

![input-2](https://github.com/BhushanMahajan25/image-synthesis-using-graphcut/blob/main/input/akeyboard_small.gif?raw=true) | ![output-2](https://github.com/BhushanMahajan25/image-synthesis-using-graphcut/blob/main/output/out-akeyboard_small.jpeg?raw=true)

## Implementation 
### Libraries used
* `imageio`==2.18.0: To perform read and write operations on images.
* `cv2`==4.5.5: To perform scaling operations.
* `numpy`==1.22.3: To effectively deal with 2d and 3d matrices.
* `matplotlib`==3.5.1: To plot graphs.
* `networkx`==2.8: To create graphs from the overlapped pixels.
### Development environment
* System: Apple Macbook Air (Apple Silicon-M1)
* Programming language: Python==3.9.8
* Text editor: MS-VS Code
### Project directories
* input: Directory of input images. User sees these input images on command line menu.
* intermediate: Directory containing images of intermediate states of graph changes, intermediate snapshots of adjacency matrices.
* output: Directory in which final synthesized image is stored.
* src: Directory containing python source code.
* screenshots: Directory containing command-line outputs as well as intermediate outputs
* requrements.txt: File containing dependencies required for the file
  
## Graphcut description
### Ford-Fulkerson algorithm
* The algorithm depends on 2 main concepts:
    1. Residual network
    2. Augmenting paths
* The graph is created from the original network with same set of vertices and one or two edges for each edge in original network.
* Augmenting path which is a path from source to sink is chosen randomely in residual network.
* The resultant flow is then appended to the current flow, and new residual network is generated.
* This process is repeated until there is no augmenting path is left.
* The running time of the Ford-Fulkerson Algorithm is $O(∣E∣ \cdot f)$ where $|E|$ is the number of edges and $f$ is the maximum flow.

## Edmonds-Karp algorithm
* Edmonds-Karp is identical to Ford-Fulkerson except for one very important trait. The search order of augmenting paths is well defined.
* Edmonds-Karp differs from Ford-Fulkerson in that it chooses the next augmenting path using breadth-first search (bfs). So, if there are multiple augmenting paths to choose from, Edmonds-Karp will be sure to choose the shortest augmenting path from the source to the sink.
* A residual network is generated from the given graph in which both the networks have same set of vertices.
* After finding this augmenting path, the cost of the path is added to the reverse edges and a new residual graph is created.
This process continues until we maximize the flow through the network i.e no augmenting paths exist.
* Edmonds-Karp improves the runtime of Ford-Fulkerson, which is $O(|E| \cdot f^{*})$ to $O(|V| \cdot |E|^{2})$ where |E| is the number of edges and |V| is the number of vertices. This improvement is important because it makes the runtime of Edmonds-Karp independent of the maximum flow of the network, $f^{*}$.

## Converting problem/solution
For this project, I have used  `networkx` library to generate the graph for overlap region between two patches.
### Intermediate steps
The below graphs indicate the edges between the nodes that ultimately sum up to the cost of minimum cut as per the intermediate steps. Edmonds-Karp algorithm is implemented on the overlapped seams which returns two set of nodes. One being the left side of the image and the other being right one. The edges plotted below represent the edges that have been cut and sum up to min cut while applying the algorithm.
 
![Figure-1](https://github.com/BhushanMahajan25/image-synthesis-using-graphcut/blob/main/intermediate/Figure_1.png?raw=true)
![Figure-2](https://github.com/BhushanMahajan25/image-synthesis-using-graphcut/blob/main/intermediate/Figure_2.png?raw=true)
![Figure-3](https://github.com/BhushanMahajan25/image-synthesis-using-graphcut/blob/main/intermediate/Figure_3.png?raw=true)
![Figure-4](https://github.com/BhushanMahajan25/image-synthesis-using-graphcut/blob/main/intermediate/Figure_4.png?raw=true)
![Figure-5](https://github.com/BhushanMahajan25/image-synthesis-using-graphcut/blob/main/intermediate/Figure_5.png?raw=true)
![Figure-6](https://github.com/BhushanMahajan25/image-synthesis-using-graphcut/blob/main/intermediate/Figure_6.png?raw=true)

There were around 60 intermediate graphs generated in the overall process out of which I have picked the above intermediate steps which clearly indicate a trend while generating the min-cut. I believe the initial density of the graph can be associated to the inclusion of pixels from both the images as we move forward. In the process, more and more pixels from the right hand side which is associated to the sink contribute to the seam.

## Generating the correct graph
The adjacency matrix has been stored in `intermediate/intermediate_adjacency_matrix.txt`

## Converting solution to image cut
The vector of cuts is stored in the file named `intermediate/intermediate_vector_cutset.txt`

## Compositing new image into old one
The above image of out_strawberry.jpeg and akeyboard_small.jpeg is the resultant file generated from this project.

## References
- https://www.csee.umbc.edu/~adamb/641/resources/GraphcutTextures.pdf
- https://www.cc.gatech.edu/cpl/projects/graphcuttextures/spring06/cos226/lectures/maxflow.pdf
- https://stackoverflow.com/questions/4482986/how-can-i-find-the-minimum-cut-on-a-graph-using-a-maximum-flow-algorithm
- https://vision.cs.uwaterloo.ca/code/
- https://github.com/networkx/networkx/blob/main/networkx/algorithms/flow/utils.py
- https://github.com/ErictheSam/Graphcut
- https://github.com/networkx/networkx/blob/main/networkx/algorithms/flow/edmondskarp.py
- https://github.com/textureguy/KUVA
- https://networkx.org/documentation/stable/reference/generated/networkx.linalg.graphmatrix.adjacency_matrix.html
- https://networkx.org/documentation/stable/reference/introduction.html
- https://www.cs.princeton.edu/courses/archive/