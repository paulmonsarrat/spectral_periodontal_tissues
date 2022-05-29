# Loading function
import glob 
import os.path
from PIL import Image
import scipy
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import scale,StandardScaler
from sklearn.decomposition import PCA

fontsize = 28
width = 21

# This function allows to browse a folder to get all the files of the spectral cube
def listdirectory(path): 
    fichier=[] 
    l = glob.glob(path+'\\*') 
    for i in l: 
        if os.path.isdir(i): fichier.extend(listdirectory(i)) 
        else: fichier.append(i) 
    return fichier

# Loading exposure images into a list
def loading_cube(path,n_transforms,rotate=0,sigma=0): 
    
    # All files of the spectral cube are loaded
    img_list = listdirectory(path)
    img_table = [[]]*len(img_list)
    img_colorcube = [[]]*len(img_list)
    count = 0
    
    for x in img_list:
        img_table[count] = [[]]*n_transforms
        img = ndimage.rotate(plt.imread(x), rotate)
        img_table[count][0] = img/255
        
        # Regular expression to affect a filename to an acquisition of the spectral cube
        if "green" in x:
            img_colorcube[count] = "Green"
        elif "red" in x:
            img_colorcube[count] = "Red"
        elif "nir" in x:
            img_colorcube[count] = "NIR"
        elif "uv" in x:
            img_colorcube[count] = "Blue"

        #Each pre-processing step is listed here, new methods may be implemented
        clahe = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(10,10))
        img_table[count][1] = scipy.ndimage.median_filter(scale(img,axis=1), sigma)
        img_table[count][2] = scipy.ndimage.median_filter(clahe.apply(img), sigma)/255
        count += 1
        
    print('Size of the spectral cube : ',len(img_table))
    print(img_colorcube)
    return img_table,img_list,img_colorcube

def compute_pca(img_table,transforms,type_transform,color_order,ncomps=3): 
    n_b = len(color_order)
    ratio = img_table[0][0].shape[0]/img_table[0][0].shape[1]
    MB_matrix = np.zeros((n_b,img_table[0][0].shape[0]*img_table[0][0].shape[1]))
    for i in range(n_b):
        MB_matrix[i,:] = img_table[i][transforms.index(type_transform)].flatten()
        #PCA is performed here
        pca = PCA(ncomps)
        pca.fit_transform(MB_matrix)
        
        # Rearranging 1D arrays to 2D arrays of image size, and back transform between 0 and 255
        PC_2d = np.zeros((img_table[0][0].shape[0],img_table[0][0].shape[1],n_b))
        for i in range(ncomps):
            PC_2d[:,:,i] = pca.components_[i].reshape(img_table[0][0].shape[0],img_table[0][0].shape[1])

        # Normalizing between 0 to 255
        PC_2d_Norm = np.zeros((img_table[0][0].shape[0],img_table[0][0].shape[1],n_b))
        for i in range(ncomps):
            PC_2d_Norm[:,:,i] = cv2.normalize(PC_2d[:,:,i],np.zeros(img_table[0][0].shape),0,255 ,cv2.NORM_MINMAX).astype(np.uint8)
    return pca, PC_2d_Norm, ratio
            
# Plot all images after transform
def plot_after_transform(transforms,img_table,save_name,img_colorcube,order,color_combine=None,burn_scale=(False,0,0,0,'white')): 
    ratio = img_table[0][0].shape[0]/img_table[0][0].shape[1]
    n_t = len(transforms)
    n_b = len(img_colorcube)
    if color_combine is not None:
        s2 = width//(n_b+1)
        s1 = s2*ratio
        fig, ax = plt.subplots(n_t,n_b+1,figsize=(s2*(n_b+1),s1*n_t),gridspec_kw=dict(hspace=0.05, wspace=0.05))
    else:
        s2 = width//n_b
        s1 = s2*ratio
        fig, ax = plt.subplots(n_t,n_b,figsize=(s2*n_b,s1*n_t),gridspec_kw=dict(hspace=0.05, wspace=0.05))
    for j in range(n_t):
        for i in range(n_b+1):
            if (i < n_b):
                ax[j,i].imshow(img_table[img_colorcube.index(order[i])][j],interpolation='none', cmap='bone')
                ax[j,0].set_ylabel(transforms[j],fontsize=fontsize)
                ax[0,i].set_title(order[i],fontsize=fontsize)
                
            if ((i <= n_b)&(color_combine is not None))|((i < n_b)&(color_combine is None)):
                ax[j,i].grid(False)
                ax[j,i].set_xticks([])
                ax[j,i].set_yticks([])
                ax[j,i].set_xticklabels([])
                ax[j,i].set_yticklabels([])
                for position in ['top','bottom','left','right']:
                    ax[0,i].spines[position].set_visible(False)
                            
        if ((color_combine is not None) & (i == n_b)):
            current_canal = [img_table[img_colorcube.index(cn)][j] for cn in color_combine[0]]
            if len(current_canal)==1:
                R = current_canal[0]
            else:
                R = np.maximum(*current_canal)
            R = cv2.normalize(R,np.zeros(R.shape),0,255,cv2.NORM_MINMAX).astype(np.uint8).reshape((R.shape[0],R.shape[1],1))
            
            current_canal = [img_table[img_colorcube.index(cn)][j] for cn in color_combine[1]]
            if len(current_canal)==1:
                G = current_canal[0]
            else:
                G = np.maximum(*current_canal)
            G = cv2.normalize(G,np.zeros(G.shape),0,255,cv2.NORM_MINMAX).astype(np.uint8).reshape((G.shape[0],G.shape[1],1))
            
            current_canal = [img_table[img_colorcube.index(cn)][j] for cn in color_combine[2]]
            if len(current_canal)==1:
                B = current_canal[0]
            else:
                B = np.maximum(*current_canal)
            B = cv2.normalize(B,np.zeros(B.shape),0,255,cv2.NORM_MINMAX).astype(np.uint8).reshape((B.shape[0],B.shape[1],1)) 
            
            current_canal = [img_table[img_colorcube.index(cn)][j] for cn in color_combine[3]]
            if len(current_canal)==1:
                M = current_canal[0]
            else:
                M = np.maximum(*current_canal)
            M = cv2.normalize(M,np.zeros(M.shape),0,255,cv2.NORM_MINMAX).astype(np.uint8).reshape((M.shape[0],M.shape[1],1))  

            rgb = np.dstack((R,G,B)).astype(np.uint8)
            m = np.dstack((M,np.zeros(M.shape),M)).astype(np.uint8)
            rgbm = cv2.addWeighted(rgb,0.5,m,0.5,0)

            ax[j,len(order)].imshow(rgbm,interpolation='none')
            ax[0,len(order)].set_title('Merge',fontsize=fontsize)

    if (burn_scale[0]):
        ax[0,0].hlines(y=int(img_table[0][0].shape[0]*burn_scale[1]),xmin=int(img_table[0][0].shape[1]*burn_scale[2]),xmax=int(img_table[0][0].shape[1]*burn_scale[2])+burn_scale[3],linewidth=2,color=burn_scale[4],linestyle='-')
    plt.savefig(save_name+'_PP.tiff', bbox_inches='tight', dpi=300)
    
def plot_after_pca(transforms,PC_2d_Norm,j,save_name,ratio=1,ncomps=3,color_combine=None,burn_scale=(False,0,0,0,'white')): 
    ratio = PC_2d_Norm[:,:,0].shape[0]/PC_2d_Norm[:,:,0].shape[1]
    n_t = len(transforms)
    if color_combine:
        s2 = width//(ncomps+1)
        s1 = s2*ratio
        fig, ax = plt.subplots(1,ncomps+1,figsize=(s2*(ncomps+1),s1),gridspec_kw=dict(hspace=0.02, wspace=0.02))
    else:
        s2 = width//(ncomps)
        s1 = s2*ratio
        fig, ax = plt.subplots(1,ncomps,figsize=(s2*ncomps,s1),gridspec_kw=dict(hspace=0.02, wspace=0.02))
    for i in range(ncomps+1):
        if (i < ncomps):
            ax[i].imshow(PC_2d_Norm[:,:,i],cmap='bone')
            #ax[i].set_title(' ',fontsize=fontsize)
            ax[i].set_title('PC'+str(i+1),fontsize=fontsize)
            ax[0].set_ylabel(transforms[j],fontsize=fontsize)

        if ((i <= ncomps)&(color_combine))|((i < ncomps)&(color_combine)):
            ax[i].grid(False)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            for position in ['top','bottom','left','right']:
                ax[i].spines[position].set_visible(False)

            rgb = np.dstack((PC_2d_Norm[:,:,0],PC_2d_Norm[:,:,1],PC_2d_Norm[:,:,2])).astype(np.uint8)
            ax[ncomps].imshow(rgb,interpolation='none')
            ax[ncomps].set_title('Merge',fontsize=fontsize)
            #ax[ncomps].set_title(' ',fontsize=fontsize)
    
    if (burn_scale[0]):
        ax[0].hlines(y=int(PC_2d_Norm[:,:,0].shape[0]*burn_scale[1]),xmin=int(PC_2d_Norm[:,:,0].shape[1]*burn_scale[2]),xmax=int(PC_2d_Norm[:,:,0].shape[1]*burn_scale[2])+burn_scale[3],linewidth=2,color=burn_scale[4],linestyle='-')
    
    plt.savefig(save_name+'_PCA_'+str(j)+'.tiff', bbox_inches='tight', dpi=300)
    
def draw_spectral_pc(path,n_transforms,rotate,sigma,transforms,type_transform,color_order,ncomps,nPC,burn_scale,save_name):
    img_table, img_list,img_colorcube = loading_cube(path,n_transforms,rotate,sigma)
    pca, PC_2d_Norm, ratio = compute_pca(img_table,transforms,type_transform,color_order,ncomps)
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.imshow(PC_2d_Norm[:,:,nPC-1],cmap='bone')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for position in ['top','bottom','left','right']:
        ax.spines[position].set_visible(False)
    ax.hlines(y=int(PC_2d_Norm[:,:,0].shape[0]*burn_scale[1]),xmin=int(PC_2d_Norm[:,:,0].shape[1]*burn_scale[2]),xmax=int(PC_2d_Norm[:,:,0].shape[1]*burn_scale[2])+burn_scale[3],linewidth=2,color=burn_scale[4],linestyle='-')
    plt.savefig(save_name+'.tiff', bbox_inches='tight', dpi=300)

def draw_spectral_merge(path,n_transforms,rotate,sigma,transforms,type_transform,color_order,ncomps,burn_scale,save_name):
    img_table, img_list,img_colorcube = loading_cube(path,n_transforms,rotate,sigma)
    pca, PC_2d_Norm, ratio = compute_pca(img_table,transforms,type_transform,color_order,ncomps)
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    rgb = np.dstack((PC_2d_Norm[:,:,0],PC_2d_Norm[:,:,1],PC_2d_Norm[:,:,2])).astype(np.uint8)
    ax.imshow(rgb,interpolation='none')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for position in ['top','bottom','left','right']:
        ax.spines[position].set_visible(False)
    ax.hlines(y=int(PC_2d_Norm[:,:,0].shape[0]*burn_scale[1]),xmin=int(PC_2d_Norm[:,:,0].shape[1]*burn_scale[2]),xmax=int(PC_2d_Norm[:,:,0].shape[1]*burn_scale[2])+burn_scale[3],linewidth=2,color=burn_scale[4],linestyle='-')
    plt.savefig(save_name+'.tiff', bbox_inches='tight', dpi=300)
