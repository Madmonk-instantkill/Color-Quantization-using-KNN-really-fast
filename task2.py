#!/usr/bin/env python
# coding: utf-8

# In[33]:


import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[34]:


A=np.array([[5.9 ,3.2],[4.6, 2.9],[6.2, 2.8],
[4.7 ,3.2],
[5.5 ,4.2],
[5.0, 3.0],
[4.9, 3.1],
[6.7 ,3.1],
[5.1, 3.8],
[6.0 ,3.0]])

red,green,blue=(6.2, 3.2),(6.6, 3.7),(6.5, 3.0)

R,B,G=[],[],[]


# # no iterations

# In[35]:


plt.figure(figsize=(15,8))
#plt.savefig("task2 iter1 a.jpg")
for i in range(0,A.shape[0]):
    r,g,b=0,0,0
    
    r,g,b=np.linalg.norm(red-A[i]),np.linalg.norm(green-A[i]),np.linalg.norm(blue-A[i])
    if r < g and r < b:
        plt.scatter(A[i][0],A[i][1],marker="^",s=170,facecolors="r",edgecolor="y")
        val1,val2=str(A[i][0]),str(A[i][1])
        strn="("+val1+","+val2+")"
        plt.text(A[i][0],A[i][1],strn)
        R.append(A[i])
        
    elif g<r and g<b:
        plt.scatter(A[i][0],A[i][1],marker="^",s=170,facecolors="g",edgecolor="y")
        val1,val2=str(A[i][0]),str(A[i][1])
        strn="("+val1+","+val2+")"
        plt.text(A[i][0],A[i][1],strn)
        G.append(A[i])
        
    elif b<r and b<g:
        plt.scatter(A[i][0],A[i][1],marker="^",s=170,facecolors="b",edgecolor="y")
        val1,val2=str(A[i][0]),str(A[i][1])
        strn="("+val1+","+val2+")"
        plt.text(A[i][0],A[i][1],strn)
        B.append(A[i])

plt.scatter(red[0],red[1],s=300,facecolors="r",edgecolor="y",label="red-mean")
val1,val2=str(red[0]),str(red[1])
strn="("+val1+","+val2+")"
plt.text(red[0],red[1],strn)

plt.scatter(blue[0],blue[1],s=300,facecolors="b",edgecolor="y",label="blue-mean")
val1,val2=str(blue[0]),str(blue[1])
strn="("+val1+","+val2+")"
plt.text(blue[0],blue[1],strn)

plt.scatter(green[0],green[1],s=300,facecolors="g",edgecolor="y",label="green-mean")
val1,val2=str(green[0]),str(green[1])
strn="("+val1+","+val2+")"
plt.text(green[0],green[1],strn)


plt.xlabel("X-cords -->")
plt.ylabel("Y-cords -->")
plt.gca().set_facecolor("yellow")
plt.legend()
plt.savefig("task2_iter1_a.jpg")

#print(red,R)
#print("--------")
#print(blue,B)
#print("---------")
#print(green,G)
#print("----------")      


# # 1st iteration

# In[36]:


red,blue,green=np.mean(np.asarray(R),axis=0),np.mean(np.asarray(B),axis=0),np.mean(np.asarray(G),axis=0)

red,blue,green


# In[37]:


plt.figure(figsize=(15,8))
R.clear()
B.clear()
G.clear()
plt.figure(figsize=(15,8))

for i in range(0,A.shape[0]):
    r,g,b=0,0,0
    r,g,b=np.linalg.norm(red-A[i]),np.linalg.norm(green-A[i]),np.linalg.norm(blue-A[i])
    if r < g and r < b:
        plt.scatter(A[i][0],A[i][1],marker="^",s=170,facecolors="w",edgecolor="k")
        val1,val2=str(A[i][0]),str(A[i][1])
        strn="("+val1+","+val2+")"
        plt.text(A[i][0],A[i][1],strn)
        R.append(A[i])
    elif g<r and g<b:
        plt.scatter(A[i][0],A[i][1],marker="^",s=170,facecolors="w",edgecolor="k")
        val1,val2=str(A[i][0]),str(A[i][1])
        strn="("+val1+","+val2+")"
        plt.text(A[i][0],A[i][1],strn)
        G.append(A[i])
    elif b<r and b<g:
        plt.scatter(A[i][0],A[i][1],marker="^",s=170,facecolors="w",edgecolor="k")
        val1,val2=str(A[i][0]),str(A[i][1])
        strn="("+val1+","+val2+")"
        plt.text(A[i][0],A[i][1],strn)
        B.append(A[i])
        
plt.scatter(red[0],red[1],s=300,facecolors="r",edgecolor="y",label="red-mean")
val1,val2=str(red[0]),str(red[1])
strn="("+val1+","+val2+")"
plt.text(red[0],red[1],strn)

plt.scatter(blue[0],blue[1],s=300,facecolors="b",edgecolor="y",label="blue-mean")
val1,val2=str(blue[0]),str(blue[1])
strn="("+val1+","+val2+")"
plt.text(blue[0],blue[1],strn)

plt.scatter(green[0],green[1],s=300,facecolors="g",edgecolor="y",label="green-mean")
val1,val2=str(green[0]),str(green[1])
strn="("+val1+","+val2+")"
plt.text(green[0],green[1],strn)

plt.xlabel("X-cords -->")
plt.ylabel("Y-cords -->")
plt.gca().set_facecolor("yellow")
plt.legend()
plt.savefig("task2_iter1_b.jpg")


#print(red,R)
#print("--------")
#print(blue,B)
#print("---------")
#print(green,G)
#print("----------")


# In[38]:


plt.figure(figsize=(15,8))
R.clear()
B.clear()
G.clear()
plt.figure(figsize=(15,8))

for i in range(0,A.shape[0]):
    r,g,b=0,0,0
    r,g,b=np.linalg.norm(red-A[i]),np.linalg.norm(green-A[i]),np.linalg.norm(blue-A[i])
    if r < g and r < b:
        plt.scatter(A[i][0],A[i][1],marker="^",s=170,facecolors="r",edgecolor="y")
        val1,val2=str(A[i][0]),str(A[i][1])
        strn="("+val1+","+val2+")"
        plt.text(A[i][0],A[i][1],strn)
        R.append(A[i])
    elif g<r and g<b:
        plt.scatter(A[i][0],A[i][1],marker="^",s=170,facecolors="g",edgecolor="y")
        val1,val2=str(A[i][0]),str(A[i][1])
        strn="("+val1+","+val2+")"
        plt.text(A[i][0],A[i][1],strn)
        G.append(A[i])
    elif b<r and b<g:
        plt.scatter(A[i][0],A[i][1],marker="^",s=170,facecolors="b",edgecolor="y")
        val1,val2=str(A[i][0]),str(A[i][1])
        strn="("+val1+","+val2+")"
        plt.text(A[i][0],A[i][1],strn)
        B.append(A[i])
        
plt.scatter(red[0],red[1],s=300,facecolors="r",edgecolor="y",label="red-mean")
val1,val2=str(red[0]),str(red[1])
strn="("+val1+","+val2+")"
plt.text(red[0],red[1],strn)

plt.scatter(blue[0],blue[1],s=300,facecolors="b",edgecolor="y",label="blue-mean")
val1,val2=str(blue[0]),str(blue[1])
strn="("+val1+","+val2+")"
plt.text(blue[0],blue[1],strn)

plt.scatter(green[0],green[1],s=300,facecolors="g",edgecolor="y",label="green-mean")
val1,val2=str(green[0]),str(green[1])
strn="("+val1+","+val2+")"
plt.text(green[0],green[1],strn)

plt.xlabel("X-cords -->")
plt.ylabel("Y-cords -->")
plt.gca().set_facecolor("yellow")
plt.legend()
plt.savefig("task2_iter2_a.jpg")


#print(red,R)
#print("--------")
#print(blue,B)
#print("---------")
#print(green,G)
#print("----------")


# # 2nd iteration

# In[39]:


red,blue,green=np.mean(np.asarray(R),axis=0),np.mean(np.asarray(B),axis=0),np.mean(np.asarray(G),axis=0)

red,blue,green


# In[40]:


plt.figure(figsize=(15,8))
R.clear()
B.clear()
G.clear()
plt.figure(figsize=(15,8))

for i in range(0,A.shape[0]):
    r,g,b=0,0,0
    r,g,b=np.linalg.norm(red-A[i]),np.linalg.norm(green-A[i]),np.linalg.norm(blue-A[i])
    if r < g and r < b:
        plt.scatter(A[i][0],A[i][1],marker="^",s=170,facecolors="r",edgecolor="y")
        val1,val2=str(A[i][0]),str(A[i][1])
        strn="("+val1+","+val2+")"
        plt.text(A[i][0],A[i][1],strn)
        R.append(A[i])
    elif g<r and g<b:
        plt.scatter(A[i][0],A[i][1],marker="^",s=170,facecolors="g",edgecolor="y")
        val1,val2=str(A[i][0]),str(A[i][1])
        strn="("+val1+","+val2+")"
        plt.text(A[i][0],A[i][1],strn)
        G.append(A[i])
    elif b<r and b<g:
        plt.scatter(A[i][0],A[i][1],marker="^",s=170,facecolors="b",edgecolor="y")
        val1,val2=str(A[i][0]),str(A[i][1])
        strn="("+val1+","+val2+")"
        plt.text(A[i][0],A[i][1],strn)
        
        B.append(A[i])
        
plt.scatter(red[0],red[1],s=300,facecolors="r",edgecolor="y",label="red-mean")
val1,val2=str(red[0]),str(red[1])
strn="("+val1+","+val2+")"
plt.text(red[0],red[1],strn)

plt.scatter(blue[0],blue[1],s=300,facecolors="b",edgecolor="y",label="blue-mean")
val1,val2=str(blue[0]),str(blue[1])
strn="("+val1+","+val2+")"
plt.text(blue[0],blue[1],strn)


plt.scatter(green[0],green[1],s=300,facecolors="g",edgecolor="y",label="green-mean")
val1,val2=str(green[0]),str(green[1])
strn="("+val1+","+val2+")"
plt.text(green[0],green[1],strn)

plt.xlabel("X-cords -->")
plt.ylabel("Y-cords -->")
plt.gca().set_facecolor("yellow")
plt.legend()
plt.savefig("task2_iter2_b.jpg")

#print(red,R)
#print("--------")
#print(blue,B)
#print("---------")
#print(green,G)
#print("----------")


# In[9]:


def knn(k,img):
    img1=img.flatten()
    img1=img.reshape(-1,3)
    data=img1.copy()
    length,width=data.shape[0],data.shape[1]
    updated_centroids,clusters,cluster_ids=[],[],[]
    np.random.seed(0)
    starting_centroids=data[np.random.randint(0,length,size=k),:]
    total_iterations,current_iteration=80,0
    cluster_points=np.ones(shape=(length,1))
    while(current_iteration < total_iterations):
        current_iteration +=1
        if current_iteration>1:
            starting_centroids=updated_centroids
        updated_centroids=np.zeros(shape=(1,width))
    
        cluster_points=np.ones(shape=(length,1))
    
        for i in range(0,k):
            kings=np.tile(starting_centroids[i],(length,1))
            distance=np.linalg.norm(kings-data,axis=1).reshape(-1,1)
            cluster_points=np.concatenate((cluster_points,distance),axis=1)
        cluster_points=np.delete(cluster_points,0,1)
        cluster_ids=np.argmin(cluster_points,axis=1).reshape(-1,1)
        data_temp=np.concatenate((data,cluster_ids),axis=1)
    
        for i in range(0,k):
            cluster_i=np.delete(data_temp[np.where(data_temp[:, -1] == i)],-1,axis=1)
            updated_cluster_i=np.divide(np.transpose(np.dot(cluster_i.T,np.ones(shape=(cluster_i.shape[0],1)))),len(cluster_i))
            updated_centroids=np.concatenate((updated_centroids,updated_cluster_i),axis=0)
        updated_centroids=np.delete(updated_centroids,0,0)     
        
    # changing the color of image
    
    for i in range(0,len(img1)):
        img1[i]=updated_centroids[cluster_ids[i]]
    img1=img1.reshape(img.shape)
    return img1


# In[10]:


img=cv2.imread("baboon.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
obtained=knn(3,img)
obtained=cv2.cvtColor(obtained,cv2.COLOR_BGR2RGB)
cv2.imwrite("task2_baboon_3.jpg",obtained)       


# In[11]:


img=cv2.imread("baboon.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
obtained=knn(5,img)
obtained=cv2.cvtColor(obtained,cv2.COLOR_BGR2RGB)
cv2.imwrite("task2_baboon_5.jpg",obtained)       


# In[12]:


img=cv2.imread("baboon.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
obtained=knn(10,img)
obtained=cv2.cvtColor(obtained,cv2.COLOR_BGR2RGB)
cv2.imwrite("task2_baboon_10.jpg",obtained)       


# In[13]:


img=cv2.imread("baboon.png")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
obtained=knn(20,img)
obtained=cv2.cvtColor(obtained,cv2.COLOR_BGR2RGB)
cv2.imwrite("task2_baboon_20.jpg",obtained)       


# In[ ]:



        

