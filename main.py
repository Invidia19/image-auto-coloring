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
from flask import Flask, render_template,request,url_for
import os
from werkzeug.utils import secure_filename
from Image_coloring import Image


UPLOAD_FOLDER = os.getcwd() + '\\static\\uploads\\'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/',methods=['POST','GET'])
def index():
	if request.method == 'POST':
		file = request.files['file']
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		image = Image(f"{UPLOAD_FOLDER + filename}")
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
		#			 print(x,y,color_acr[i])
					try:
						color_avl[y].pop(color_avl[y].index(color_acr[i]))
					except ValueError:
						pass

		image.set_color(color_acr)

		pallete = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[127,150,0],[150,0,127]]
		image.set_palette(pallete)


		img_rgb = np.stack((image.get_skel(),)*3, axis=-1)
		cv2.imwrite(UPLOAD_FOLDER + 'skeleton_not_colored.jpg',img_rgb)


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

		cv2.imwrite(UPLOAD_FOLDER + 'skeletoncolored.jpg',img_rgb)
		return f"<img src=\"{url_for('static',filename='uploads/skeletoncolored.jpg' )}\" >"



if __name__ == '__main__':
	app.run(debug=True)