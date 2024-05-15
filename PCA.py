# Import packages
import numpy as np
#import cupy as np
#from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import collections 
#from matplotlib import cm
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial import distance
from yellowbrick.cluster import SilhouetteVisualizer
from matplotlib.lines import Line2D
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from imutils import paths
import math
import sys
from sys import argv
import xlsxwriter
import csv
#import tensorflow as tf
#import numba
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())
# grab the image paths 
dataset =  os.path.join(sys.argv[1])

"""
For Japanese File Name included Kanji
https://qiita.com/SKYS/items/cbde3775e2143cad7455
https://qiita.com/ka10ryu1/items/5fed6b4c8f29163d0d65
"""
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None
def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


#dataset =r'e:\AI\picture\code_8\train\patch'
#dataset =r'C:\Users\SG19\Desktop\MNIST'
print("[INFO] loading images...")
print("[INFO] loading",dataset)



imagePaths = sorted(list(paths.list_images(dataset)))
W =48
H =48
WHD =W*H
K = int(sys.argv[2])
#K=2
IMAGE_DIMS = (W,H)
###############################################################################
# initialize the data and labels
data = []
labels = []

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list2Dgraph
	image = imread(imagePath, cv2.IMREAD_GRAYSCALE)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

	# extract set of class labels from the image path and update the
	# labels list
	l = label = imagePath.split(os.path.sep)[-3].split("_")
	labels.append(l)
    
    # scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))
# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

lbl_for_plot= []
# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
    classes=("{}. {}".format(i + 1, label))
    classes1=("{}".format( label))
    print(classes)
    lbl_for_plot.append(classes1)

Orglbl=[]
trainY=[]
for i in labels:
    Index=np.where(i == i.max())[0]
    Orglbl.append(int(Index))
#    trainY=np.append(int(Index))
# = labels
trainX = data



###############################################################################
trainX=trainX.reshape(-1,WHD)
X=trainX
Y=Orglbl

#X1=X[:,0:,0] #(283,301) Width
#X2=X[:,0] # (283,601) Hight
###############################################################################
#transforming input image values to 2D using PCA
pca=PCA(2).fit(X)
X_proj = pca.transform(X)

###############################################################################
#transforming input image values to 3D using PCA
pca3d=PCA(3).fit(X)
X_proj3d = pca3d.transform(X)

###############################################################################


#predicting K-Means Values via pca
k_digits = KMeans(n_clusters=K, random_state=None)
y_pcapred=k_digits.fit_predict(X_proj,Y)
#predicting the cluster centroidspoints 2D
centers1 = np.array(k_digits.cluster_centers_)
k_digits.labels_



###############################################################################
#predicting 3D K-Means Values via 3Dpca
k_digits = KMeans(n_clusters=K)
y_pcapred3D=k_digits.fit_predict(X_proj3d)
#predicting the cluster centroidspoints 3D

#predicting the cluster centroidspoints 3D
centers3D = np.array(k_digits.cluster_centers_)

###############################################################################



#counting the labels
colorg = collections.Counter(Y)
#sorting the counted labels
colorg=sorted(colorg.items())
colorg=dict(colorg)
print('INPUT----------->Label Count',colorg)



colp = collections.Counter(y_pcapred)
collection_pca_KMean = sorted(colp.items())
collection_pca_KMean=dict(collection_pca_KMean)
print((collection_pca_KMean))

xlrow = 0
xlcol = 0




C1 = []
C2 = []
C3 = []
C4 = []
C5 = []
C6 = []
C7 = []
C8 = []
C9 = []
C10 = []


for m in range(len(X)):
     if y_pcapred[m]==0:
       gx=np.array([Y[m]])
       C1.extend(gx)
       
     elif y_pcapred[m]==1 :
         gx=np.array([Y[m]])
         C2.extend(gx)
         
     elif y_pcapred[m]==2 :
         gx=np.array([Y[m]])
         C3.extend(gx)
     elif y_pcapred[m]==3 :
         gx=np.array([Y[m]])
         C4.extend(gx)
     
     elif y_pcapred[m]==4 :
         gx=np.array([Y[m]])
         C5.extend(gx)
     elif y_pcapred[m]==5 :
         gx=np.array([Y[m]])
         C6.extend(gx)
     
     elif y_pcapred[m]==6 :
         gx=np.array([Y[m]])
         C7.extend(gx)
     elif y_pcapred[m]==7 :
         gx=np.array([Y[m]])
         C8.extend(gx)
     elif y_pcapred[m]==8 :
         gx=np.array([Y[m]])
         C9.extend(gx)
     elif y_pcapred[m]==9 :
         gx=np.array([Y[m]])
         C10.extend(gx)



rowdata=[]
        


try:
    for n in range(len(collection_pca_KMean)):
       data=[[n,C1.count(n),(round((C1.count(n)/collection_pca_KMean[0])*100))]]
       rowdata.extend(data)

    for n in range(len(collection_pca_KMean)):
       data=[[n,C2.count(n),(round((C2.count(n)/collection_pca_KMean[1])*100))]]
       rowdata.extend(data)       


    for n in range(len(collection_pca_KMean)):
       data=[[n,C2.count(n),(round((C3.count(n)/collection_pca_KMean[2])*100))]]
       rowdata.extend(data)
      

    for n in range(len(collection_pca_KMean)):
       data=[[n,C4.count(n),(round(((C4.count(n)/collection_pca_KMean[3])*100)))]]
       rowdata.extend(data)


    for n in range(len(collection_pca_KMean)):
       data=[[n,C5.count(n),(round(((C5.count(n)/collection_pca_KMean[4])*100)))]]
       rowdata.extend(data)


    for n in range(len(collection_pca_KMean)):
       data=[[n,C6.count(n),(round(((C6.count(n)/collection_pca_KMean[5])*100)))]]
       rowdata.extend(data)


    for n in range(len(collection_pca_KMean)):
       data=[[n,C7.count(n),(round((C7.count(n)/collection_pca_KMean[6])*100))]]
       rowdata.extend(data)

    for n in range(len(collection_pca_KMean)):
       data=[[n,C8.count(n),(round(((C8.count(n)/collection_pca_KMean[7])*100)))]]
       rowdata.extend(data)

    for n in range(len(collection_pca_KMean)):
       data=[[n,C9.count(n),(round(((C9.count(n)/collection_pca_KMean[8])*100)))]]
       rowdata.extend(data)

    for n in range(len(collection_pca_KMean)):
       data=[[n,C10.count(n),(round(((C10.count(n)/collection_pca_KMean[9])*100)))]]
       rowdata.extend(data)       
except:
    pass

rowHead=[["判定色","判定ラベル","合計","クラスラベル","合計","割合(%)"]]
workbook = xlsxwriter.Workbook('clusterdata.xlsx')
worksheet = workbook.add_worksheet()     
worksheet.add_table('A1:F1', {'data': rowHead,'header_row': False})
worksheet.add_table('D2:F100', {'data': rowdata,'header_row': False})

#workbook.close()




rowdata1=[]
CLUSTER_COLOR= {0 : '赤', 1 :'緑' , 2 : '青',
                   3 : 'シアン', 4 : 'MAGENTA', 5 : 'YELLOW',
                   6 : 'ORANGE', 7 : 'LIGHTGREEN',
                   8 : 'BROWN', 9 : 'PURPLE'}
d=K-1
for n in range(len(collection_pca_KMean)):
    data1 =[[CLUSTER_COLOR[n],n,collection_pca_KMean[n]]]
    for i in range(d):
        data1.append([])
    rowdata1.extend(data1)

#workbook = xlsxwriter.Workbook('clusterdata1.xlsx')
#worksheet = workbook.add_worksheet()

worksheet.add_table('A2:C100', {'data': rowdata1,'header_row': False})
#worksheet.merge_range('A2:C100', 'Merged Range', rowdata1)
workbook.close()
###############################################################################

C11 = []
C21 = []
C31 = []
C41 = []
C51 = []
C61 = []
C71 = []
C81 = []
C91 = []
C101 = []


for m in range(len(X)):
     if Y[m]==0:
       gx=np.array([y_pcapred[m]])
       C11.extend(gx)
       
     elif Y[m]==1 :
         gx=np.array([y_pcapred[m]])
         C21.extend(gx)
         
     elif Y[m]==2 :
         gx=np.array([y_pcapred[m]])
         C31.extend(gx)
     elif Y[m]==3 :
         gx=np.array([y_pcapred[m]])
         C41.extend(gx)
     
     elif Y[m]==4 :
         gx=np.array([y_pcapred[m]])
         C51.extend(gx)
     elif Y[m]==5 :
         gx=np.array([y_pcapred[m]])
         C61.extend(gx)
     
     elif Y[m]==6 :
         gx=np.array([y_pcapred[m]])
         C71.extend(gx)
     elif Y[m]==7 :
         gx=np.array([y_pcapred[m]])
         C81.extend(gx)
     elif Y[m]==8 :
         gx=np.array([y_pcapred[m]])
         C91.extend(gx)
     elif Y[m]==9 :
         gx=np.array([y_pcapred[m]])
         C101.extend(gx)

rowdata=[]
        
CLUSTER_COLOR= {0 : '赤', 1 :'緑' , 2 : '青',
                   3 : 'シアン', 4 : 'MAGENTA', 5 : 'YELLOW',
                   6 : 'ORANGE', 7 : 'LIGHTGREEN',
                   8 : 'BROWN', 9 : 'PURPLE'}
blank =[[CLUSTER_COLOR[n],n,0,0]]
try:
    for n in range(K):
       data=[[CLUSTER_COLOR[n],n,C11.count(n),(round((C11.count(n)/colorg[0])*100))]] 
       rowdata.extend(data)

    for n in range(len(collection_pca_KMean)):
       data=[[CLUSTER_COLOR[n],n,C21.count(n),(round((C21.count(n)/colorg[1])*100))]]
       rowdata.extend(data)       


    for n in range(len(collection_pca_KMean)):
       data=[[CLUSTER_COLOR[n],n,C31.count(n),(round((C31.count(n)/colorg[2])*100))]]
       rowdata.extend(data)
      

    for n in range(len(collection_pca_KMean)):
       data=[[CLUSTER_COLOR[n],n,C41.count(n),(round((C41.count(n)/colorg[3])*100))]]
       rowdata.extend(data)


    for n in range(len(collection_pca_KMean)):
       data=[[CLUSTER_COLOR[n],n,C51.count(n),(round((C51.count(n)/colorg[4])*100))]]
       rowdata.extend(data)


    for n in range(len(collection_pca_KMean)):
       data=[[CLUSTER_COLOR[n],n,C61.count(n),(round((C61.count(n)/colorg[5])*100))]]
       rowdata.extend(data)


    for n in range(len(collection_pca_KMean)):
       data=[[CLUSTER_COLOR[n],n,C71.count(n),(round((C71.count(n)/colorg[6])*100))]]
       rowdata.extend(data)

    for n in range(len(collection_pca_KMean)):
       data=[[CLUSTER_COLOR[n],n,C81.count(n),(round((C81.count(n)/colorg[7])*100))]]
       rowdata.extend(data)

    for n in range(len(collection_pca_KMean)):
       data=[[CLUSTER_COLOR[n],n,C91.count(n),(round((C91.count(n)/colorg[8])*100))]]
       rowdata.extend(data)

    for n in range(len(collection_pca_KMean)):
       data=[[CLUSTER_COLOR[n],n,C101.count(n),(round((C101.count(n)/colorg[9])*100))]]
       rowdata.extend(data)       
except:
    pass

rowHead=[["クラスラベル","合計","判定","判定ラベル","合計","割合(%)"]]

workbook = xlsxwriter.Workbook('clusterdata1.xlsx')
worksheet = workbook.add_worksheet()     
worksheet.add_table('A1:F1', {'data': rowHead,'header_row': False})
worksheet.add_table('C2:F100', {'data': rowdata,'header_row': False})


rowdata1=[]

d=K-1
for n in range(len(colorg)):
    data1 =[[n,colorg[n]]]
    for i in range(d):
        data1.append([])
    rowdata1.extend(data1)



worksheet.add_table('A2:B100', {'data': rowdata1,'header_row': False})

workbook.close()
###############################################################################


LABEL_COLOR_MAP = {0 : 'r', 1 :'darkgreen' , 2 : 'b',
                   3 : 'c', 4 : 'm', 5 : 'y',
                   6 : 'orange', 7 : 'g',
                   8 : 'brown', 9 : 'purple'}

Marker = {0 : 'v', 1 : 's', 2 : 'P',
                   3 : '*', 4 : 'p', 5 : 'd',
                   6 : 'X', 7 : '^',
                   8 : '>', 9 : '<', }

cluster = {0 : 'C0', 1 : 'C1', 2 : 'C2',
                   3 : 'C3', 4 : 'C4', 5 : 'C5',
                   6 : 'C6', 7 : 'C7',
                   8 : 'C8', 9 : 'C9', }




with open("Output.txt", "w") as text_file:
    text_file.write("")
    
workbook = xlsxwriter.Workbook('PlotData.xlsx')
worksheet = workbook.add_worksheet()
row = 0
col = 0
i=0
for imagePath in imagePaths:
#    fig = plt.figure()
    j = Y[i]
    k = y_pcapred[i]
    orglbl = lbl_for_plot[j]
    if len(lbl_for_plot) == K:
        Predlbl =lbl_for_plot[k]
    else:
        Predlbl =str(list(collection_pca_KMean)[k])
#    print(imagePath,orglbl,Predlbl)

    worksheet.write(row, col, imagePath)#A
    worksheet.write(row, col+1, orglbl)#B
    worksheet.write(row, col+2, Predlbl)#C
    
    worksheet.write(row, col+3, X_proj[:,0][i])#E
    worksheet.write(row, col+4, X_proj[:,1][i])#F    
    row+=1 
    worksheet.write(0, col+5, centers1[:,0][0])
    worksheet.write(1, col+5, centers1[:,0][1])
    worksheet.write(0, col+6, centers1[:,1][0])
    worksheet.write(1, col+6, centers1[:,1][1])
    
    
#    imge = imread(imagePath)
#    plt.imshow(imge, cmap = 'gray')
#    disp =("ImageName:",os.path.basename(imagePath),"Label-",orglbl,"  Assigned_Cluster-",Predlbl)
#    plt.title(disp)
#    plt.axis('off')
#    fig.savefig(os.path.join(os.path.dirname(__file__),'PictureBox/')+(os.path.basename(imagePath)), dpi=fig.dpi)
#    print(os.path.basename(imagePath))
    with open(os.path.join(os.path.dirname(__file__),'PictureBox/')+(os.path.basename(imagePath).replace("png","txt")), "w") as f:
        f.close()
    
    with open("Output.txt", "a") as text_file:
        text_file.write("{},{},{}\n".format(os.path.basename(imagePath),orglbl,Predlbl))
    i +=1
#    plt.close("all")
workbook.close()

#For n_clusters = 2 The average silhouette_score is : 0.7049787496083262
#For n_clusters = 3 The average silhouette_score is : 0.5882004012129721
#For n_clusters = 4 The average silhouette_score is : 0.6505186632729437
#For n_clusters = 5 The average silhouette_score is : 0.56376469026194
#For n_clusters = 6 The average silhouette_score is : 0.4504666294372765
###############################################################################



###############################################################################

model = k_digits
visualizer = SilhouetteVisualizer(model)
visualizer.fit(X_proj) # Fit the training data to the visualizer    
visualizer.poof(outpath="visualizer.png") # Draw/show/poof the data
#visualizer.poof() # Draw/show/poof the data
               
# Compute the silhouette scores for avarage
sample_silhouette_values = silhouette_samples(X_proj, y_pcapred,metric='euclidean')
silhouette_avg = silhouette_score(X_proj, y_pcapred,metric='euclidean')


###############################################################################
#PCA to KMeans 3D Graphics

workbook = xlsxwriter.Workbook('3Dgraph.xlsx')
worksheet = workbook.add_worksheet()
row = 0
col = 0
crow=0
i=0
for imagePath in imagePaths:
    worksheet.write(row, col, imagePath)#A
    worksheet.write(row, col+1, Y[i])#B
    worksheet.write(row, col+2, y_pcapred[i])#C
    worksheet.write(row, col+3, X_proj3d[:,0][i])#E
    worksheet.write(row, col+4, X_proj3d[:,1][i])#F
    worksheet.write(row, col+5, X_proj3d[:,2][i])#F    
    row+=1 
    i+=1
    worksheet.write(crow, col+9, K)
for i in range(len(centers3D)):
    worksheet.write(crow, col+6, centers3D[:,0][i])
    worksheet.write(crow, col+7, centers3D[:,1][i])
    worksheet.write(crow, col+8, centers3D[:,2][i])
    crow+=1
    
workbook.close()
###############################################################################
#fig = plt.figure()
##label_color = [LABEL_COLOR_MAP[l] for l in y_pcapred3D]
#ax = Axes3D(fig)
##i=0
##for i in range(len(X)):
##
##    ax.scatter(X_proj3d[:, 0][i], X_proj3d[:, 1][i], X_proj3d[:, 2][i], c=LABEL_COLOR_MAP[Y[i]],  marker= Marker[y_pcapred[i]])
##    i +=1
#
#i=0
#for i in range(len(X)):
#    ax.scatter(X_proj3d[:,0][i], X_proj3d[:,1][i], X_proj3d[:,2][i], c='None')
#    ax.text(X_proj3d[:,0][i], X_proj3d[:,1][i],X_proj3d[:,2][i],str(lbl_for_plot[Y[i]]),color=LABEL_COLOR_MAP[y_pcapred[i]],horizontalalignment='center',verticalalignment='center',size=12)
#    i +=1
#
#
#
#
#ax.scatter(centers3D[:, 0], centers3D[:, 1], centers3D[:, 2], marker='x', c='#050505', s=1000)
#for i in range(K):
#    ax.text(centers3D[:,0][i], centers3D[:,1][i], centers3D[:,2][i],cluster[i])
#collection_pca_KMean =dict(collection_pca_KMean)
#lgnd_lbl=[]
##for i in list(collection_pca_KMean):
##    if list(collection_pca_KMean)[i] == list(LABEL_COLOR_MAP)[i]:
##        label2 =(collection_pca_KMean)[i]
##
##        label1 =cluster[i]
##        label=(str(label1),str(label2))   
##
##        color =(LABEL_COLOR_MAP)[i]
##        color =(str(color))
##        labels =(Line2D([0], [0], marker=Marker[i], color='black', label=label))
##        lgnd_lbl.append(labels)
##        i +=1
##    else:
##        pass
##    
##    
#lgnd_lbl=((Line2D([0], [0],marker='o',color='None', label="Number: Class No")),(Line2D([0], [0],marker='o',color='None',label="Color: Cluster")))
##    
##for i in range(len(lbl_for_plot)) :
##        label1 =lbl_for_plot[i]
##        label2=dict(colorg)[i]
##
##
##        label=(str(label1),str(label2))   
##        labels1=(Line2D([0], [0], marker='o', color=LABEL_COLOR_MAP[i], label=label))
##        lgnd_lbl.append(labels1)
##        i +=1
#
#ax.legend(loc='upper right',bbox_to_anchor=(1,1),handles=lgnd_lbl, fancybox=True)
#ax.set_title('PCA to K-Means', fontsize=10) 
#fig.savefig('C:/Users/SG19/Desktop/RisingStar_0001/PyScripts/plots/fig1.png', dpi=fig.dpi) 

###############################################################################
#PCA to KMean Graphs
workbook = xlsxwriter.Workbook('2Dgraph.xlsx')
worksheet = workbook.add_worksheet()
row = 0
col = 0
crow=0
i=0
for imagePath in imagePaths:
    worksheet.write(row, col, imagePath)#A
    worksheet.write(row, col+1, Y[i])#B
    worksheet.write(row, col+2, y_pcapred[i])#C
    worksheet.write(row, col+3, X_proj[:,0][i])#E
    worksheet.write(row, col+4, X_proj[:,1][i])#F   
    row+=1 
    i+=1
    worksheet.write(crow, col+9, K)
for i in range(len(centers3D)):
    worksheet.write(crow, col+6, centers1[:,0][i])
    worksheet.write(crow, col+7, centers1[:,1][i])

    crow+=1
   
workbook.close()


###############################################################################
#fig = plt.figure()
a = centers1[:,0]
b = centers1[:,1]
#
#
#
#
##MARK = [Marker[m] for m in Y_dev ]
#
#
#
##label_color = [LABEL_COLOR_MAP[l] for l in y_pcapred]
#
##i=0
##for i in range(len(X)):
##    plt.scatter(X_proj[:,0][i], X_proj[:,1][i], c=LABEL_COLOR_MAP[Y[i]],  marker= Marker[y_pcapred[i]])
##    i +=1
#
#
#i=0
#for i in range(len(X)):
#    plt.scatter(X_proj[:,0][i], X_proj[:,1][i], c='None')
#    plt.text(X_proj[:,0][i], X_proj[:,1][i],str(lbl_for_plot[Y[i]]),color=LABEL_COLOR_MAP[y_pcapred[i]],horizontalalignment='center',verticalalignment='center',size=12)
#    i +=1
#
#
#
#plt.scatter(centers1[:,0], centers1[:,1], marker="x", color='black')
#for i in range(K):
#    plt.text(centers1[:,0][i], centers1[:,1][i],cluster[i])
#plt.title('PCA to K-Means', fontsize=10) 
#plt.plot(a,b, linestyle='dotted', marker='x', color='black')
#
#collection_pca_KMean =dict(collection_pca_KMean)
#lgnd_lbl=[]
##for i in list(collection_pca_KMean):
##    if list(collection_pca_KMean)[i] == list(LABEL_COLOR_MAP)[i]:
##        label2 =(collection_pca_KMean)[i]
##
##        label1 =cluster[i]
##        label=(str(label1),str(label2))   
##
##        color =(LABEL_COLOR_MAP)[i]
##        color =(str(color))
##        labels =(Line2D([0], [0], marker=Marker[i], color='black', label=label))
##        lgnd_lbl.append(labels)
##        i +=1
##    else:
##        pass
#lgnd_lbl=((Line2D([0], [0],marker='o',color='None', label="Number: Class No")),(Line2D([0], [0],marker='o',color='None',label="Color: Cluster")))
##    
##for i in range(len(lbl_for_plot)) :
##        label1 =lbl_for_plot[i]
##        label2=dict(colorg)[i]
##
##
##        label=(str(label1),str(label2))   
##        labels1=(Line2D([0], [0], marker='o', color=LABEL_COLOR_MAP[i], label=label))
##        lgnd_lbl.append(labels1)
##        i +=1
#plt.legend(loc='upper right',bbox_to_anchor=(1,1),handles=lgnd_lbl, fancybox=True)#fig.savefig('C:/Users/SG19/Desktop/RisingStar_0001/PyScripts/plots/fig2.png', dpi=fig.dpi) 
#
#x=X_proj[:,0]
#y=X_proj[:,1]
#
#
#from PIL import Image
#def onclick(event):
#    ix, iy = event.xdata, event.ydata
##    print("I clicked at x={0:5.2f}, y={1:5.2f}".format(ix,iy))
#  
#    
#    # Calculate, based on the axis extent, a reasonable distance 
#    # from the actual point in which the click has to occur (in this case 5%)
#    ax = plt.gca()
#    dx = 0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0])
#    dy = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
#
#    # Check for every point if the click was close enough:
#    for i in range(len(x)):
#        if(x[i] > ix-dx and x[i] < ix+dx and y[i] > iy-dy and y[i] < iy+dy):
#            
#            Image.open(imagePaths[i]).show() 
#       
##            print("You clicked close enough!",i,"the Xicoordinates",x[i],"Yi coordinates",y[i])
#cid = fig.canvas.mpl_connect('button_press_event', onclick)
#
#plt.show('all')

#Euclidian Distance

avgdist = 0
for i in range (K):
    for j in range (K):
        x1 = a[i]
        x2 = a[j]
        y1 = b[i]
        y2 = b[j]
        if i==j:
            pass
        else:
            eucl=math.sqrt(((x1)-(x2))**2+((y1)-(y2))**2)
            dist = eucl
            avgdist=avgdist+(dist)
            print("distence between cluster",i,"to cluster",j,dist)
    if j == K-1:
        break
    else:
        j = j+1
    i +=1
    
sum_of_dist = 0
for i in range(len(Y)):
        if y_pcapred[i] ==0:
            eucl=math.sqrt(((centers1[:,0][0])-(X_proj[:,0][i]))**2+((centers1[:,1][0])-(X_proj[:,0][i]))**2)
            sum_of_dist += eucl
        elif y_pcapred[i] == 1:
            eucl=math.sqrt(((centers1[:,0][1])-(X_proj[:,0][i]))**2+((centers1[:,1][1])-(X_proj[:,0][i]))**2)
            sum_of_dist += eucl    
        elif y_pcapred[i] == 2:
            eucl=math.sqrt(((centers1[:,0][2])-(X_proj[:,0][i]))**2+((centers1[:,1][2])-(X_proj[:,0][i]))**2)
            sum_of_dist += eucl 
        elif y_pcapred[i] == 3:
            eucl=math.sqrt(((centers1[:,0][3])-(X_proj[:,0][i]))**2+((centers1[:,1][3])-(X_proj[:,0][i]))**2)
            sum_of_dist += eucl 
        elif y_pcapred[i] == 4:
            eucl=math.sqrt(((centers1[:,0][4])-(X_proj[:,0][i]))**2+((centers1[:,1][4])-(X_proj[:,0][i]))**2)
            sum_of_dist += eucl 
        elif y_pcapred[i] == 5:
            eucl=math.sqrt(((centers1[:,0][5])-(X_proj[:,0][i]))**2+((centers1[:,1][5])-(X_proj[:,0][i]))**2)
            sum_of_dist += eucl 
        elif y_pcapred[i] == 6:
            eucl=math.sqrt(((centers1[:,0][6])-(X_proj[:,0][i]))**2+((centers1[:,1][6])-(X_proj[:,0][i]))**2)
            sum_of_dist += eucl 
        elif y_pcapred[i] == 7:
            eucl=math.sqrt(((centers1[:,0][7])-(X_proj[:,0][i]))**2+((centers1[:,1][7])-(X_proj[:,0][i]))**2)
            sum_of_dist += eucl 
        elif y_pcapred[i] == 8:
            eucl=math.sqrt(((centers1[:,0][8])-(X_proj[:,0][i]))**2+((centers1[:,1][8])-(X_proj[:,0][i]))**2)
            sum_of_dist += eucl 
        elif y_pcapred[i] == 9:
            eucl=math.sqrt(((centers1[:,0][9])-(X_proj[:,0][i]))**2+((centers1[:,1][9])-(X_proj[:,0][i]))**2)
            sum_of_dist += eucl     

avarage_cluster_distance = (avgdist)/(K-1) 
print(round(avarage_cluster_distance,3))  

avarage_distance_between_samples= sum_of_dist/len(Y)
print(round(avarage_distance_between_samples,3))

print(round(silhouette_avg,3))
