import cv2
from cv2 import cvtColor
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import data, filters, color, morphology
from skimage.segmentation import flood, flood_fill
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert


class Image():
    def __init__(self,filename):
        self.filename = filename

    def set_color(self,color):
        self._color = color

    def get_color(self):
        return self._color

    def set_palette(self,palette):
        self._palette = palette

    def get_palette(self):
        return self._palette

    def set_edge(self,dict_connect):
        self._edge = dict_connect

    def get_edge(self):
        return self._edge

    def set_image(self,img):
        self._img = img

    def get_image(self):
        return self._img

    def set_skel(self,img):
        self._skel_img = img

    def get_skel(self):
        return self._skel_img

    def get_neighbor_and_edge_pixel(self,i,j,r):
        ori_img = self._skel_img
        c = self._skel_img[i][j]
        neighbor = [(i,j)]
        taboo = [(i,j)]
        edge = []
        neig_graph = []
        relev_pixel = []

        while len(neighbor) > 0:
            if self._skel_img[neighbor[0][0]][neighbor[0][1]] == c:
                self._skel_img[neighbor[0][0]][neighbor[0][1]] = r
                if neighbor[0][0] - 1 >= 0:
                    if self._skel_img[neighbor[0][0]-1][neighbor[0][1]] == c or self._skel_img[neighbor[0][0]-1][neighbor[0][1]] == 255:
                        neighbor.append((neighbor[0][0]-1,neighbor[0][1]))
                        taboo.append((neighbor[0][0]-1,neighbor[0][1]))

                if neighbor[0][0] + 1 < self._skel_img.shape[0]:   
                    if self._skel_img[neighbor[0][0]+1][neighbor[0][1]] == c or self._skel_img[neighbor[0][0]+1][neighbor[0][1]] == 255:
                        neighbor.append((neighbor[0][0]+1,neighbor[0][1]))
                        taboo.append((neighbor[0][0]+1,neighbor[0][1]))

                if neighbor[0][1] - 1 >= 0:            
                    if self._skel_img[neighbor[0][0]][neighbor[0][1]-1] == c or self._skel_img[neighbor[0][0]][neighbor[0][1]-1] == 255:
                        neighbor.append((neighbor[0][0],neighbor[0][1]-1))
                        taboo.append((neighbor[0][0],neighbor[0][1]-1))

                if neighbor[0][1] + 1 < self._skel_img.shape[1]:   
                    if self._skel_img[neighbor[0][0]][neighbor[0][1]+1] == c or self._skel_img[neighbor[0][0]][neighbor[0][1]+1] == 255:
                        neighbor.append((neighbor[0][0],neighbor[0][1]+1))
                        taboo.append((neighbor[0][0],neighbor[0][1]+1))

            elif self._skel_img[neighbor[0][0]][neighbor[0][1]] == 255:
                edge.append((neighbor[0][0],neighbor[0][1]))
            neighbor.pop(0)
        
        for index in edge:
            min_i = max(index[0]-1,0)
            min_j = max(index[1]-1,0)
            max_i = min(index[0]+1,self._skel_img.shape[0]-1)
            max_j = min(index[1]+1,self._skel_img.shape[1]-1)
            for i in range(min_i,max_i+1):
                for j in range(min_j,max_j+1):
                    if self._skel_img[i][j] != r and self._skel_img[i][j] != 255:
                        neig_graph.append((i,j))
                    if self._skel_img[i][j] == r and self._skel_img[i][j] != 255:
                        relev_pixel.append((i,j))
                    
        self._skel_img = ori_img
        return neig_graph, relev_pixel

    def get_nodes(self):
        return self._nodes

    def set_nodes(self):
        ori_img = self._skel_img
        node_img = []

        for i,x in enumerate(self._skel_img):
            for j,y in enumerate(x):
                if self._skel_img[i][j] == 0:
                    node_img.append((i,j))
                    self._skel_img = flood_fill(self._skel_img, (i,j), 127,connectivity=1)

        self._skel_img = ori_img
        self._nodes =  node_img

    def flood_fill_rgb(self,img,i,j,r):
        c = np.zeros_like(img[i][j])
        c[:] = img[i][j][:]
        neighbor = [(i,j)]
        while len(neighbor) > 0:
            
            if (img[neighbor[0][0]][neighbor[0][1]] == c).all():
                img[neighbor[0][0]][neighbor[0][1]][:] = r[:]
                if neighbor[0][0] - 1 >= 0:
                    if (img[neighbor[0][0]-1][neighbor[0][1]] == c).all():
                        neighbor.append((neighbor[0][0]-1,neighbor[0][1]))
                if neighbor[0][0] + 1 < img.shape[0]:   
                    if (img[neighbor[0][0]+1][neighbor[0][1]] == c).all():
                        neighbor.append((neighbor[0][0]+1,neighbor[0][1]))
                if neighbor[0][1] - 1 >= 0:            
                    if (img[neighbor[0][0]][neighbor[0][1]-1] == c).all():
                        neighbor.append((neighbor[0][0],neighbor[0][1]-1))
                if neighbor[0][1] + 1 < img.shape[1]:   
                    if (img[neighbor[0][0]][neighbor[0][1]+1] == c).all():
                        neighbor.append((neighbor[0][0],neighbor[0][1]+1)) 
            neighbor.pop(0)


def main():
    image = Image(input("Masukkan file gambar:"))
    image.set_image(cv2.imread(image.filename,cv2.IMREAD_GRAYSCALE)) 

    ret,thresh1 = cv2.threshold(image.get_image(),127,255,cv2.THRESH_BINARY)

    image.set_skel(skeletonize(invert(thresh1)/255)*255)
    image.set_nodes()

    neigs = {}
    homes = {}
    node_img = image.get_nodes()
    for i,j in node_img:
        if (i,j) not in neigs.keys():
            neigs[(i,j)] = []
        neig,home= image.get_neighbor_and_edge_pixel(i,j,127)
        neigs[(i,j)].append(neig)
        neigs[(i,j)].append(home)



    dict_connect = {}
    for x in neigs.items():
        for y in neigs.items():
            if x[0] not in dict_connect.keys():
                dict_connect[x[0]] = {}
            if x[0] != y[0]:
                dict_connect[x[0]][y[0]] = False


    for nodes in neigs.items():
        node_neig = nodes[1][0]
        node_home = nodes[1][1]
        for neig in node_neig:
            for node_s in neigs.items():
                node_s_neig = node_s[1][0]
                node_s_home = node_s[1][1]
                if nodes[0] != node_s[0] and not(dict_connect[nodes[0]][node_s[0]]) and neig in node_s_home:
                    dict_connect[nodes[0]][node_s[0]] = True

    image.set_edge(dict_connect)

    color_avl = {}
    for y in image.get_edge(): 
        color_avl[y] =[x for x in range(len(image.get_edge()))]


    color_acr = [0 for x in range(len(image.get_edge()))]


    for i,x in enumerate(image.get_edge()):
        color_acr[i] = color_avl[x][0]
        for j,y in enumerate(image.get_edge()[x]):        
            if image.get_edge()[x][y]:
    #             print(x,y,color_acr[i])
                try:
                    color_avl[y].pop(color_avl[y].index(color_acr[i]))
                except ValueError:
                    pass

    image.set_color(color_acr)

    pallete = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[127,150,0],[150,0,127]]
    image.set_palette(pallete)


    img_rgb = np.stack((image.get_skel(),)*3, axis=-1)
    cv2.imwrite('skeleton_not_colored.jpg',img_rgb)


    for index,(i,j) in enumerate(image.get_nodes()):
        image.flood_fill_rgb(img_rgb,i,j,np.array(image.get_palette()[image.get_color()[index]]))


    ori_img_rgb = np.stack((image.get_image(),)*3, axis=-1)
    ret,thresh1 = cv2.threshold(ori_img_rgb,127,255,cv2.THRESH_BINARY)


    for i,x in enumerate(thresh1):
        for j,y in enumerate(x):
            if (y == np.array([0,0,0])).all():
                img_rgb[i][j][:] = y[:]


    img_rgb= np.array(img_rgb, dtype=np.uint8) 
    img_rgb = cvtColor(img_rgb,cv2.COLOR_RGB2BGR)

    cv2.imwrite('skeletoncolored.jpg',img_rgb)

if __name__ == '__main__':
    main()