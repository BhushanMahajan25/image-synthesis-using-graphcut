import config
import os
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from copy import deepcopy
from edmondsKarp import edmonds_karp

fig_index = 1

class Mincut(object):
    def __init__(self,input_image,output_image_rows,output_image_cols):
        self.overlap_cols = 0
        self.overlap_rows = 0
        self.input_image = input_image
        self.variance = np.var(input_image)
        input_image_shape = input_image.shape
        self.patch_rows = input_image_shape[0]
        self.patch_cols = input_image_shape[1]
        self.original_rows = output_image_rows
        self.original_cols = output_image_cols
        output_image_rows += self.patch_rows
        output_image_cols += self.patch_cols
        self.output_image_rows = output_image_rows
        self.output_image_cols = output_image_cols
        self.old_graph = np.zeros((output_image_rows,output_image_cols,3),dtype = np.int)
        self.new_graph = np.zeros((output_image_rows,output_image_cols,3),dtype = np.int)
        self.mask = np.zeros((output_image_rows,output_image_cols),dtype = np.int)
        self.overlap_area = np.zeros((output_image_rows,output_image_cols),dtype = np.int)
        self.pixcel_seam = np.zeros((output_image_rows,output_image_cols,2),dtype = np.int)
        self.seam_init_val = np.zeros((output_image_rows,output_image_cols,2),dtype = np.int)
        self.corner_mask = [self.output_image_rows,0,self.output_image_rows,0]
        self.index = 1
    
    def reform_mask(self,mask_coord):
        max_i = min(mask_coord[0] + self.patch_rows, self.output_image_rows)
        max_j = min(mask_coord[1] + self.patch_cols, self.output_image_cols)
        self.mask[mask_coord[0]:max_i,mask_coord[1]:max_j] = np.ones((max_i-mask_coord[0],max_j-mask_coord[1]),dtype=np.int)
        if (mask_coord[0] < self.corner_mask[0]):
            self.corner_mask[0] = mask_coord[0]
        if(mask_coord[0]+self.patch_rows > self.corner_mask[1]):
            self.corner_mask[1] = mask_coord[0]+self.patch_rows
        if(mask_coord[1] < self.corner_mask[2] ):
            self.corner_mask[2] = mask_coord[1]
        if(mask_coord[1]+self.patch_cols > self.corner_mask[3]):
            self.corner_mask[3] = mask_coord[1]+self.patch_cols

    def reform_seam_init_val(self,corner_overlap,corner,mask_seam):
        for x in range(self.patch_rows):
            for y in range(self.patch_cols):
                x_coord = corner[0] + x
                y_coord = corner[1] + y
                if(x_coord >= corner_overlap[0] and x_coord < corner_overlap[0]+self.overlap_rows and y_coord >= corner_overlap[1] and y_coord < corner_overlap[1]+self.overlap_cols):
                    if(mask_seam[x_coord-corner_overlap[0]][y_coord-corner_overlap[1]] == 2):
                        self.seam_init_val[x_coord][y_coord][0] = x
                        self.seam_init_val[x_coord][y_coord][1] = y
                else:
                    self.seam_init_val[x_coord][y_coord][0] = x
                    self.seam_init_val[x_coord][y_coord][1] = y

    def initialize(self):
        mask_coord = [0,0]
        self.old_graph[mask_coord[0]:mask_coord[0]+self.patch_rows,mask_coord[1]:mask_coord[1]+self.patch_cols] = self.input_image[0:self.patch_rows,0:self.patch_cols]
        self.new_graph= deepcopy(self.old_graph)
        self.reform_mask(mask_coord)
        self.reform_seam_init_val(mask_coord,mask_coord,np.ones((self.patch_rows,self.patch_cols)))

    def get_mask_neighbors(self,curr_vertices):
        neighbors = 0
        for i in range(-1,2):
            for j in range(-1,2):
                if(curr_vertices[0]+i >= 0 and curr_vertices[0] < self.output_image_rows and curr_vertices[1]+j >= 0 and curr_vertices[1]+j < self.output_image_cols):
                    if(self.mask[curr_vertices[0]+i][curr_vertices[1]+j] == 0):
                        neighbors += 1
        return neighbors

    def get_overlap_neighbors(self,curr_vertices):
        neighbors = 0
        for i in range(-1,2):
            for j in range(-1,2):
                if(curr_vertices[0]+i >= 0 and curr_vertices[0] < self.output_image_rows and curr_vertices[1]+j >= 0 and curr_vertices[1]+j < self.output_image_cols):
                    if(self.overlap_area[curr_vertices[0]+i][curr_vertices[1]+j] == 0):
                        neighbors += 1
        return neighbors

    def reform_overlap_area(self,curr_coord):
        self.overlap_area = np.zeros((self.output_image_rows,self.output_image_cols),dtype=np.int)
        corner_coord = [0,0]
        self.overlap_rows = self.overlap_cols = 0
        first = True
        n = 0
        for x in range(self.patch_rows):
            for y in range(self.patch_cols):
                if(self.mask[curr_coord[0]+x][curr_coord[1]+y] == 1): 
                    self.overlap_area[curr_coord[0]+x][curr_coord[1]+y] = 1
                    if(first):
                        corner_coord[0] = curr_coord[0]+x
                        corner_coord[1] = curr_coord[1]+y
                        first = False
                    if(n == 0):
                        self.overlap_rows += 1
                    n += 1
            if(n != 0 and n > self.overlap_cols):
                self.overlap_cols = n
            n = 0
        for x in range(corner_coord[0]-1,corner_coord[0]+self.overlap_rows):
            for y in range(corner_coord[1]-1,corner_coord[1]+self.overlap_cols):
                if(self.overlap_area[x][y] == 1):
                    if(self.get_overlap_neighbors([x,y]) >= 1):
                        if(self.get_mask_neighbors([x,y]) >= 1):
                            self.overlap_area[x][y] = 2
                        else:
                            self.overlap_area[x][y] = 3
        return corner_coord

    def reform_seam(self,corner,mask_seam,patch_index):
        found = False
        for x in range(self.overlap_rows):
            for y in range(self.overlap_cols):
                self.pixcel_seam[corner[0]+x][corner[1]+y][0] = 0
                self.pixcel_seam[corner[0]+x][corner[1]+y][1] = 1
                found = False
                mask_val = mask_seam[x][y]
                if(x < mask_seam.shape[0] - 1 and found == False):
                    if(mask_seam[x+1][y] != mask_val and mask_seam[x+1][y] != 0):
                        self.pixcel_seam[corner[0]+x][corner[1]+y][1] = 2
                        if(mask_val == 2):
                            self.pixcel_seam[corner[0]+x][corner[1]+y][0] = patch_index
                        elif(mask_val == 1 and self.pixcel_seam[corner[0]+x][corner[1]+y][0] == 0):
                            self.pixcel_seam[corner[0]+x][corner[1]+y][0] = 1
                        found = True
                if(y < mask_seam.shape[1] - 1 and found == False):
                    if(mask_seam[x][y+1] != mask_val and mask_seam[x][y+1] != 0):
                        if(self.pixcel_seam[corner[0]+x][corner[1]+y][1] == 2):
                            self.pixcel_seam[corner[0]+x][corner[1]+y][1] = 3
                        else:
                            self.pixcel_seam[corner[0]+x][corner[1]+y][1] = 4
                        if(mask_val == 2):
                            self.pixcel_seam[corner[0]+x][corner[1]+y][0] = patch_index
                        elif(mask_val == 1 and self.pixcel_seam[corner[0]+x][corner[1]+y][0] == 0):
                            self.pixcel_seam[corner[0]+x][corner[1]+y][0] = 1
                        found = True

    def match_entire_patch(self):
        patching = np.zeros((self.original_rows,self.original_cols),dtype = np.float)
        tk = np.zeros((self.original_cols, self.patch_rows, self.patch_cols, 3),dtype = np.float)
        mask = np.zeros((self.original_cols,self.patch_rows, self.patch_cols))
        si = np.zeros((self.original_rows, self.original_cols))
        for x in range(self.original_rows):
            w1 = min(self.patch_rows,self.original_rows-x)
            for y in range(self.original_cols):
                w2 = min(self.patch_cols,self.original_cols-y)
                mask[y][:] = np.zeros((self.patch_rows, self.patch_cols))
                a1 = self.input_image[0:w1,0:w2]
                a2 = self.old_graph[x:x+w1,y:y+w2]
                tk[y][0:w1,0:w2] = ( a1 -a2 )
                mask[y][0:w1,0:w2] = self.mask[x:x+w1,y:y+w2]
                si[x][y] = w1 * w2
            for k in range(3):
                patching[x] += np.sum( (tk[:,:,:,k]*mask ) ** 2,axis = (1,2))
        patching /= si
        patching = np.exp(-patching/ (0.3 * self.variance) ).reshape(-1)
        patching /= np.sum(patching)
        l = int(np.random.choice( self.original_rows * self.original_cols ,1,p = patching))
        patch_coord = [0,0]
        patch_coord[0] = int(l/int(self.original_cols))
        patch_coord[1] = int(l - self.original_cols * patch_coord[0])
        return patch_coord

    def calculate_edge_cost(self,x_coord,y_coord,x_adj,y_adj,A,B):
        new_coord = B[x_coord][y_coord]
        old_coord = A[x_coord][y_coord]
        new_adj = B[x_adj][y_adj]
        old_adj = A[x_adj][y_adj]
        r = abs(old_coord[0]-new_coord[0])+abs(old_adj[0]-new_adj[0])
        g = abs(old_coord[1]-new_coord[1])+abs(old_adj[1]-new_adj[1])
        b = abs(old_coord[2]-new_coord[2])+abs(old_adj[2]-new_adj[2])
        return (r+g+b)/3

    def perform_mincut(self,curr_coord):
        new_graph_pixels = [0]
        # Copying the 3d ip_image at the new random co-ordinate in new_graph
        self.new_graph[curr_coord[0]:curr_coord[0]+self.patch_rows,curr_coord[1]:curr_coord[1]+self.patch_cols] = self.input_image[0:self.patch_rows,0:self.patch_cols]
        overlap_corner = self.reform_overlap_area(curr_coord)
        new_graph_pixels[0] = np.sum(self.mask[curr_coord[0]:curr_coord[0]+self.patch_rows,curr_coord[1]:curr_coord[1]+self.patch_cols])
        G = nx.Graph()

        mask_seam = np.zeros((self.overlap_rows, self.overlap_cols ),dtype = np.int)
        n = 2
        seam_support = 0
        matrix2D = np.zeros((self.overlap_rows, self.overlap_cols ),dtype = np.int)

        for i in range(self.overlap_rows):
            for j in range(self.overlap_cols):
                if(self.mask[overlap_corner[0]+i][overlap_corner[1]+j]==1):
                    
                    matrix2D[i][j] = n
                    n += 1
        n = 2

        for x in range(self.overlap_rows):
            for y in range(self.overlap_cols):
                x_coord = overlap_corner[0]+x
                y_coord = overlap_corner[1]+y

                down = False
                right = False

                if(self.mask[x_coord][y_coord]== 1):

                    if(self.pixcel_seam[x_coord][y_coord][0] != 0):
                        if(self.pixcel_seam[x_coord][y_coord][1] == 2 or self.pixcel_seam[x_coord][y_coord][1] == 3):
                            if(self.overlap_area[x_coord][y_coord] == 1 and self.mask[x_coord+1][y_coord] == 1):
                                down = True
                                seam_support += 1                                
                                source_As = self.seam_init_val[x_coord][y_coord]
                                sink_As = source_As + np.array([1,0])
                                sink_At = self.seam_init_val[x_coord+1][y_coord]
                                source_At = sink_At - np.array([1,0])

                                r = abs(self.input_image[source_As[0]][source_As[1]][0]-self.input_image[source_At[0]][source_At[1]][0])+abs(self.input_image[sink_As[0]][sink_As[1]][0]-self.input_image[sink_At[0]][sink_At[1]][0])
                                g = abs(self.input_image[source_As[0]][source_As[1]][1]-self.input_image[source_At[0]][source_At[1]][1])+abs(self.input_image[sink_As[0]][sink_As[1]][1]-self.input_image[sink_At[0]][sink_At[1]][1])
                                b = abs(self.input_image[source_As[0]][source_As[1]][2]-self.input_image[source_At[0]][source_At[1]][2])+abs(self.input_image[sink_As[0]][sink_As[1]][2]-self.input_image[sink_At[0]][sink_At[1]][2])
                                cost = (r+g+b)/3
                                G.add_edge(0,new_graph_pixels[0]+1+seam_support,capacity=cost)

                                r = abs(self.input_image[source_As[0]][source_As[1]][0]-self.new_graph[x_coord][y_coord][0])+abs(self.input_image[sink_As[0]][sink_As[1]][0]-self.new_graph[x_coord+1][y_coord][0])
                                g = abs(self.input_image[source_As[0]][source_As[1]][1]-self.new_graph[x_coord][y_coord][1])+abs(self.input_image[sink_As[0]][sink_As[1]][1]-self.new_graph[x_coord+1][y_coord][1])
                                b = abs(self.input_image[source_As[0]][source_As[1]][2]-self.new_graph[x_coord][y_coord][2])+abs(self.input_image[sink_As[0]][sink_As[1]][2]-self.new_graph[x_coord+1][y_coord][2])
                                cost = (r+g+b)/3
                                G.add_edge(matrix2D[x][y],new_graph_pixels[0]+1+seam_support, capacity=cost)

                                r = abs(self.new_graph[x_coord][y_coord][0]-self.input_image[source_At[0]][source_At[1]][0])+abs(self.new_graph[x_coord+1][y_coord][0]-self.input_image[sink_At[0]][sink_At[1]][0])
                                g = abs(self.new_graph[x_coord][y_coord][1]-self.input_image[source_At[0]][source_At[1]][1])+abs(self.new_graph[x_coord+1][y_coord][1]-self.input_image[sink_At[0]][sink_At[1]][1])
                                b = abs(self.new_graph[x_coord][y_coord][2]-self.input_image[source_At[0]][source_At[1]][2])+abs(self.new_graph[x_coord+1][y_coord][2]-self.input_image[sink_At[0]][sink_At[1]][2])
                                cost = (r+g+b)/3
                                G.add_edge(matrix2D[x+1][y],new_graph_pixels[0]+1+seam_support, capacity=cost)

                        if(self.pixcel_seam[x_coord][y_coord][1] == 4 or self.pixcel_seam[x_coord][y_coord][1] == 3):
                            if(self.overlap_area[x_coord][y_coord] == 1 and self.mask[x_coord][y_coord+1] == 1):
                                right = True
                                seam_support += 1
                                source_As = self.seam_init_val[x_coord][y_coord]
                                sink_As = source_As + np.array([0,1])
                                sink_At = self.seam_init_val[x_coord][y_coord+1]
                                source_At = sink_At - np.array([0,1])

                                r = abs(self.input_image[source_As[0]][source_As[1]][0]-self.input_image[source_At[0]][source_At[1]][0])+abs(self.input_image[sink_As[0]][sink_As[1]][0]-self.input_image[sink_At[0]][sink_At[1]][0])
                                g = abs(self.input_image[source_As[0]][source_As[1]][1]-self.input_image[source_At[0]][source_At[1]][1])+abs(self.input_image[sink_As[0]][sink_As[1]][1]-self.input_image[sink_At[0]][sink_At[1]][1])
                                b = abs(self.input_image[source_As[0]][source_As[1]][2]-self.input_image[source_At[0]][source_At[1]][2])+abs(self.input_image[sink_As[0]][sink_As[1]][2]-self.input_image[sink_At[0]][sink_At[1]][2])
                                cost = (r+g+b)/3
                                G.add_edge(new_graph_pixels[0]+1+seam_support,1, capacity=cost)

                                r = abs(self.input_image[source_As[0]][source_As[1]][0]-self.new_graph[x_coord][y_coord][0])+abs(self.input_image[sink_As[0]][sink_As[1]][0]-self.new_graph[x_coord][y_coord+1][0])
                                g = abs(self.input_image[source_As[0]][source_As[1]][1]-self.new_graph[x_coord][y_coord][1])+abs(self.input_image[sink_As[0]][sink_As[1]][1]-self.new_graph[x_coord][y_coord+1][1])
                                b = abs(self.input_image[source_As[0]][source_As[1]][2]-self.new_graph[x_coord][y_coord][2])+abs(self.input_image[sink_As[0]][sink_As[1]][2]-self.new_graph[x_coord][y_coord+1][2])
                                cost = (r+g+b)/3
                                G.add_edge(matrix2D[x][y],new_graph_pixels[0]+1+seam_support, capacity=cost)

                                r = abs(self.new_graph[x_coord][y_coord][0]-self.input_image[source_At[0]][source_At[1]][0])+abs(self.new_graph[x_coord][y_coord+1][0]-self.input_image[sink_At[0]][sink_At[1]][0])
                                g = abs(self.new_graph[x_coord][y_coord][1]-self.input_image[source_At[0]][source_At[1]][1])+abs(self.new_graph[x_coord][y_coord+1][1]-self.input_image[sink_At[0]][sink_At[1]][1])
                                b = abs(self.new_graph[x_coord][y_coord][2]-self.input_image[source_At[0]][source_At[1]][2])+abs(self.new_graph[x_coord][y_coord+1][2]-self.input_image[sink_At[0]][sink_At[1]][2])
                                cost = (r+g+b)/3
                                G.add_edge(matrix2D[x][y+1],new_graph_pixels[0]+1+seam_support, capacity=cost)

                    if(x < self.overlap_rows-1 and self.mask[x_coord+1][y_coord] == 1 and down == False ):
                        x_adj = x_coord + 1
                        y_adj = y_coord
                        cost = self.calculate_edge_cost(x_coord,y_coord,x_adj,y_adj,self.old_graph,self.new_graph)
                        G.add_edge(matrix2D[x][y],matrix2D[x+1][y],capacity=cost)

                    if(y < self.overlap_cols-1 and self.mask[x_coord][y_coord+1] == 1 and right == False ):
                        x_adj = x_coord
                        y_adj = y_coord + 1
                        
                        cost = self.calculate_edge_cost(x_coord,y_coord,x_adj,y_adj,self.old_graph,self.new_graph)
                        G.add_edge(matrix2D[x][y],matrix2D[x][y+1],capacity=cost)

                    if(self.overlap_area[x_coord][y_coord] == 2):
                        G.add_edge(matrix2D[x][y],1,capacity=1<<20)
                    if(self.overlap_area[x_coord][y_coord] == 3):
                        G.add_edge(0,matrix2D[x][y],capacity=1<<20)
    
        mincut_val, partition = nx.minimum_cut(G,0,1,flow_func=edmonds_karp)
        # mincut_val, partition = nx.minimum_cut(G,0,1)

        left, right = partition
        cut_set = set()
        for u, nbrs in ((n, G[n]) for n in left):
            cut_set.update((u, v) for v in nbrs if v in right)
        # print(sorted(cut_set))
        # temp_graph=nx.DiGraph()
        # temp_graph.add_edges_from(cut_set)
        # nx.draw_networkx(temp_graph)
        # plt.show() 
        file = open(os.path.join(config.INTERMEDIATE_DIR ,"intermediate_cutset.txt"),"a")
        file.write(str(cut_set))
        file.write("\n")
        file.close()
        adj_matrix = nx.adjacency_matrix(G)
        file = open(os.path.join(config.INTERMEDIATE_DIR, "intermediate_adjacency_matrix.txt"),"a")
        file.write(str(adj_matrix))
        file.write("\n")
        file.close()

        # # saving intermediate op in file

        # global fig_index
        # fig = plt.gcf()
        # fig.set_size_inches((11, 8.5), forward=False)
        # fig.savefig(os.path.join(config.INTERMEDIATE_DIR, "figure-{}.png".format(fig_index)), format="PNG", dpi=500)
        # fig_index += 1

        l = [0 for i in range(G.number_of_nodes())]
        for i in partition[1]:
            l[i] = 1
        for i in range(self.overlap_rows):
            for j in range(self.overlap_cols):
                x_coord = overlap_corner[0] + i
                y_coord = overlap_corner[1] + j
                if(self.overlap_area[x_coord][y_coord] != 0):
                    if( l[matrix2D[i][j]] == l[0]):
                        self.new_graph[x_coord][y_coord] = self.old_graph[x_coord][y_coord]
                        mask_seam[i][j] = 1
                    else:
                        mask_seam[i][j] = 2
        self.reform_seam(overlap_corner,mask_seam,self.index)
        self.reform_seam_init_val(overlap_corner,curr_coord,mask_seam)

        self.old_graph = deepcopy(self.new_graph)
        self.reform_mask(curr_coord)
        self.overlap_area = np.zeros((self.output_image_rows,self.output_image_cols),dtype=np.int)
        self.index += 1
    
    def patch(self):
        self.initialize()
        x = random.randint( int(1*self.patch_rows/3), int(2*self.patch_rows/3))
        y = 0
        while(y < self.original_cols):
            while(x < self.original_rows):
                self.perform_mincut([x,y])
                x += random.randint( int(1*self.patch_rows/3), int(2*self.patch_rows/3))
            y += random.randint( int(2*self.patch_cols/3), int(2*self.patch_cols/3))
            x = 0
        for _ in range(5):
            res_coord = self.match_entire_patch()
            self.perform_mincut(res_coord)
        return self.new_graph[0:self.original_rows,0:self.original_cols]
