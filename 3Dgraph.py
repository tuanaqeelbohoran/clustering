# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:27:25 2019

@author: sg19
"""

import xlrd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
file_location = "3Dgraph.xlsx"
workbook = xlrd.open_workbook(file_location)
first_sheet = workbook.sheet_by_index(0)

imgpath = [first_sheet.cell_value(i, 0) for i in range(first_sheet.nrows)]
Y = [first_sheet.cell_value(i, 1) for i in range(first_sheet.nrows)]
y_pcapred = [first_sheet.cell_value(i, 2) for i in range(first_sheet.nrows)]
X_proj3d_0 = [first_sheet.cell_value(i, 3) for i in range(first_sheet.nrows)]
X_proj3d_1 = [first_sheet.cell_value(i, 4) for i in range(first_sheet.nrows)]
X_proj3d_2 = [first_sheet.cell_value(i, 5) for i in range(first_sheet.nrows)]

centers3D_0= [first_sheet.cell_value(i, 6) for i in range(first_sheet.nrows)]
centers3D_1= [first_sheet.cell_value(i, 7) for i in range(first_sheet.nrows)]
centers3D_2= [first_sheet.cell_value(i, 8) for i in range(first_sheet.nrows)]

K= first_sheet.cell_value(0, 9) 

#centers3D_0=np.array(centers3D_0)
#centers3D_1=np.array(centers3D_1)
#centers3D_2=np.array(centers3D_2)


centers3D=[centers3D_0,centers3D_1,centers3D_2]

X_proj3d = [X_proj3d_0, X_proj3d_1,X_proj3d_2]

LABEL_COLOR_MAP = {0 : 'r', 1 :'darkgreen' , 2 : 'b',
                   3 : 'c', 4 : 'm', 5 : 'y',
                   6 : 'orange', 7 : 'g',
                   8 : 'brown', 9 : 'purple'}

cluster = {0 : 'C0', 1 : 'C1', 2 : 'C2',
                   3 : 'C3', 4 : 'C4', 5 : 'C5',
                   6 : 'C6', 7 : 'C7',
                   8 : 'C8', 9 : 'C9', }

fig = plt.figure()
ax = Axes3D(fig)

i=0
for i in range(len(Y)):
    ax.scatter(X_proj3d_0[i], X_proj3d_1[i], X_proj3d_2[i], c='None')
    ax.text(X_proj3d_0[i], X_proj3d_1[i], X_proj3d_2[i],str(int((Y[i]))),color=LABEL_COLOR_MAP[y_pcapred[i]],horizontalalignment='center',verticalalignment='center',size=12)
    i +=1





for i in range(int(K)):
    ax.scatter(centers3D_0[i], centers3D_1[i],centers3D_2[i], marker='x', c='#050505', s=1000)
    ax.text(centers3D_0[i], centers3D_1[i],centers3D_2[i],cluster[i])
#collection_pca_KMean =dict(collection_pca_KMean)
lgnd_lbl=[]
#    
lgnd_lbl=((Line2D([0], [0],marker='o',color='None', label="Number: Class No")),(Line2D([0], [0],marker='o',color='None',label="Color: Cluster")))

ax.legend(loc='upper right',bbox_to_anchor=(1,1),handles=lgnd_lbl, fancybox=True)
ax.set_title('PCA to K-Means', fontsize=10) 






#x=X_proj3d_0
#y=X_proj3d_0
#z=X_proj3d_0
#from PIL import Image
#
#def onclick(event):
#    ix, iy = event.xdata, event.ydata
#    ax = plt.gca()
#    dx = 0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0])
#    dy = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
#    
#
#    
##     Check for every point if the click was close enough:
#    for i in range(len(x)):
#        if(x[i] > ix-dx and x[i] < ix+dx and y[i] > iy-dy and y[i] < iy+dy):
#         print(event)  
##                Image.open(imgpath[i]).show() 
#       
#cid = fig.canvas.mpl_connect('button_press_event', onclick)


plt.show('all')






















plt.show('all')