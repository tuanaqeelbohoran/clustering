# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:06:54 2019

@author: sg19
"""

import xlrd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from PIL import Image
file_location = "2Dgraph.xlsx"
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)

imagePaths = [first_sheet.cell_value(i, 0) for i in range(first_sheet.nrows)]
Y = [first_sheet.cell_value(i, 1) for i in range(first_sheet.nrows)]
y_pcapred = [first_sheet.cell_value(i, 2) for i in range(first_sheet.nrows)]
X_proj_0 = [first_sheet.cell_value(i, 3) for i in range(first_sheet.nrows)]
X_proj_1 = [first_sheet.cell_value(i, 4) for i in range(first_sheet.nrows)]





K= first_sheet.cell_value(0, 9) 

centers1_0= [first_sheet.cell_value(i, 6) for i in range(int(K))]
centers1_1= [first_sheet.cell_value(i, 7) for i in range(int(K))]
#print(centers1_0)
#centers3D_0=np.array(centers3D_0)
#centers3D_1=np.array(centers3D_1)
#centers3D_2=np.array(centers3D_2)


centers1=[centers1_0,centers1_1]

X_proj = [X_proj_0, X_proj_1]

LABEL_COLOR_MAP = {0 : 'r', 1 :'darkgreen' , 2 : 'b',
                   3 : 'c', 4 : 'm', 5 : 'y',
                   6 : 'orange', 7 : 'g',
                   8 : 'brown', 9 : 'purple'}

cluster = {0 : 'C0', 1 : 'C1', 2 : 'C2',
                   3 : 'C3', 4 : 'C4', 5 : 'C5',
                   6 : 'C6', 7 : 'C7',
                   8 : 'C8', 9 : 'C9', }



fig = plt.figure()
a = centers1_0
b = centers1_1



i=0
for i in range(len(Y)):
    plt.scatter(X_proj_0[i], X_proj_1[i], c='None')
    plt.text(X_proj_0[i], X_proj_1[i],str(int(Y[i])),color=LABEL_COLOR_MAP[y_pcapred[i]],horizontalalignment='center',verticalalignment='center',size=12)
    i +=1




for i in range(int(K)):
    plt.scatter(centers1_0[i], centers1_1[i], marker="x", color='black')
    plt.text(centers1_0[i], centers1_1[i],cluster[i])
plt.title('PCA to K-Means', fontsize=10) 
plt.plot(a,b, linestyle='dotted', marker='x', color='black')

#collection_pca_KMean =dict(collection_pca_KMean)
lgnd_lbl=[]

lgnd_lbl=((Line2D([0], [0],marker='o',color='None', label="Number: Class No")),(Line2D([0], [0],marker='o',color='None',label="Color: Cluster")))

plt.legend(loc='upper right',bbox_to_anchor=(1,1),handles=lgnd_lbl, fancybox=True)#fig.savefig('C:/Users/SG19/Desktop/RisingStar_0001/PyScripts/plots/fig2.png', dpi=fig.dpi) 

x=X_proj_0
y=X_proj_1


#from PIL import Image
def onclick(event):
    ix, iy = event.xdata, event.ydata
    ax = plt.gca()
    dx = 0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    dy = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])

    # Check for every point if the click was close enough:
    for i in range(len(x)):
        if(x[i] > ix-dx and x[i] < ix+dx and y[i] > iy-dy and y[i] < iy+dy):
            
            Image.open(imagePaths[i]).show() 
       
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show('all')