__author__ = 'mimighostipad'

import numpy as np
import random
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

def get_start_points(points_num,x_max = 1000,y_max = 1000):
    points = list()
    for i in range(points_num):
        x = random.random()*x_max
        y = random.random()*y_max
        points.append((x,y))
    return points

def gaussian_sample(x,y,sigma = 20):
    new_x = np.random.normal(x,sigma)
    new_y = np.random.normal(y,sigma)
    return new_x,new_y

def generate_clusters(start_points_num, points_num):

    centroids = get_start_points(start_points_num)

    clusters = defaultdict(list)

    for i in range(points_num):
        center = random.choice(centroids)
        new_point = gaussian_sample(center[0],center[1])
        clusters[center].append(new_point)

    return centroids,clusters

centroids,clusters = generate_clusters(4,500)


f = open("centroids.txt",'w')

for centroid in centroids:
    f.write(str(centroid[0]) + ","+str(centroid[1])+'\n')


all_points = []

for center in clusters:
    all_points += clusters[center]

f2 = open("points.txt",'w')
for point in all_points:
    f2.write(str(point[0])+","+str(point[1])+'\n')


print all_points

x = [i[0] for i in all_points]
y = [i[1] for i in all_points]

plt.scatter(x,y)

plt.show()