# image-auto-coloring
Application of graph coloring for image coloring.


How to run:
  python main.py

Input:
a genering black and white image used for coloring, example:<br/>
![sample](/static/uploads/batik.jpg)

Output:
a random colored image, example:<br/>
![sample](/static/uploads/skeletoncolored.jpg)

Improvements needed
1. Algorithm for determining neighbour for each node is still very slow
2. Thinning algorithm used, shorten some edges example:<br/>
![sample](/static/uploads/example-bad-thinning.JPG)
this then leads to a problem where the 2 nodes merges into one
3. Input color-preferences




