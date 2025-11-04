import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import streamlit
import uuid
import os
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as st
from SVHC import new_cor, pre_cor, inner_product, distance_fisher, kullback_leibler
from collections import defaultdict,OrderedDict
import cv2
import scipy as sp
from matplotlib import gridspec
from astropy.io import fits


def plot_images_grid_arrays(image_type, star, images, bands, colorscale="Hot"):
    n_images = len(images)
    if n_images == 0:
        raise ValueError("No images to display.")
    
    n_cols = math.ceil(math.sqrt(n_images))
    n_rows = math.ceil(n_images / n_cols)
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[bands[i] if i < len(bands) else f"Image {i+1}" for i in range(n_images)]
    )
    
    for idx, img in enumerate(images):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1
        
        # Heatmap per le immagini (0-1)
        fig.add_trace(
            go.Heatmap(z=img, colorscale=colorscale, showscale=False),
            row=row, col=col
        )
        
        # Nascondi assi
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)
        
        # Mantieni aspect ratio quadrato
        fig.update_yaxes(scaleanchor=f"x{idx+1}", row=row, col=col)

    fig.update_layout(
        height=250*n_rows, width=250*n_cols,
        showlegend=False,
        title_text=f"{image_type}, {star}"
    )
    
    streamlit.plotly_chart(fig,width = "stretch",key = str(uuid.uuid4()))
    
    

def ToMemb(nod,N):
    h = np.array([-1]*N)

    nod = sorted(nod,key=lambda x:len(x),reverse=True)
    Hn = []
    sel = {}
    while True:
        c = 0
        for i in range(len(nod)):
            if i in sel: continue
            if not all(h[np.array(nod[i])]==-1): continue

            h[np.array(nod[i])]=c
            sel[i] = None
            c+=1

        h[h==-1] = np.arange(h.max()+1,(h==-1).sum()+h.max()+1)
        Hn.append(h)
        h = np.array([-1]*N)
        if len(sel)==len(nod):
            break

    Hn.append(np.arange(N))    
    return Hn


# In[3]:


def StandardDendrogram(H):
    h = sorted(list(zip(*H)))
    h = list(map(list,h))

    for j in range(len(h[0])-1):
        k = OrderedDict.fromkeys(list(zip(*h))[j]).keys()
        k = dict(zip(k,range(len(k))))

        for i in range(len(h)):
            h[i][j] = k[h[i][j]]

    j = len(h[0])-1

    k = OrderedDict.fromkeys(list(zip(*h))[j]).keys()
    k = dict(zip(k,range(len(k))))

    h = sorted(h,key=lambda x:x[-1])

    for i in range(len(h)):
            h[i][j] = k[h[i][j]]

    h = list(map(np.array,zip(*h)))
    return h


# In[4]:


def OrderArray(H):
    return np.array( list(zip(*sorted(zip(H[-1],range(len(H[0]))))))[1] )


# In[5]:


def DendroAndCorrDists(x, star, image_type, labels, validated, dendrogram,
                       method='average', xylabel='Objects',
                       file_w=None, color_style='Set1', nan=False, metric='pearson', 
                       tau_m=50, size_tick=0.02, title_size=10, fontsize_labels=8):
    
    ordn = OrderArray(StandardDendrogram(ToMemb(dendrogram.keys(), x.shape[0])))
    x = x[ordn]
    labels_ord = list(labels[ordn])
    
    rev_ornd = dict(sorted(zip(ordn, range(len(ordn)))))
    dendrogram = {tuple(sorted([rev_ornd[k] for k in K])):(np.array(sorted([rev_ornd[a] for a in dendrogram[K][0]])),
                                                             np.array(sorted([rev_ornd[a] for a in dendrogram[K][1]]))) 
                  for K in dendrogram}
    validated = [tuple(sorted([rev_ornd[k] for k in K])) for K in validated]

    if not nan:
        if metric=='pearson':
            R = 1 - np.corrcoef(x)
        elif metric=='spearman':
            R = 1 - st.spearmanr(x.T)[0]
        elif metric == "new_cor":
            Z, autocor = pre_cor(x, tau_m)
            rho = new_cor(x, Z, autocor, tau_m)
            R= 1. - rho
        elif metric == 'inner_product':
            R = inner_product(x)
        elif metric == 'kullback_leibler':
            R = kullback_leibler(x)

    validated = sorted(validated, key=len)
    N = R.shape[0]
    K = dendrogram.keys()
    nc  = len(validated)
    cnmx = sns.color_palette(color_style, nc-1)
    col = defaultdict(int)
    for k in K:
        for j in range(len(validated)):
            if set(k) <= set(validated[j]):
                if tuple(validated[j])==tuple(range(N)):
                    col[tuple(k)] = 'black'
                    break
                col[tuple(k)] = cnmx[j-1]
                break

    # FIGURA LARGA PER 50 LABEL
    fig = plt.figure(figsize=(max(12, N*0.25), 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.4])
    axarr = [plt.subplot(gs[0]), plt.subplot(gs[1])]

    pos = defaultdict(tuple)
    for i in range(N):
        pos[(i,)] = (float(i), 0)

    K = sorted(dendrogram.keys(), key=len)
    hm = []
    pp_tick=[0]
    for k in K:
        d = tuple(sorted(dendrogram[k][0])), tuple(sorted(dendrogram[k][1]))
        c = tuple(sorted(k))
        pp = pos[d[0]][0], pos[d[1]][0]
        pp_tick.append(pp[0])
        h0 = pos[d[0]][1], pos[d[1]][1]
        if method=='average':
            h = R[dendrogram[k][0]][:,dendrogram[k][1]].mean()
        elif method=='complete':
            h = R[dendrogram[k][0]][:,dendrogram[k][1]].max()
        elif method=='single':
            h = R[dendrogram[k][0]][:,dendrogram[k][1]].min()
        axarr[0].plot([pp[0],pp[0]],[h0[0],h],'-',color='k',alpha=0.5)
        axarr[0].plot([pp[1],pp[1]],[h0[1],h],'-',color='k',alpha=0.5)
        axarr[0].plot([pp[0],pp[1]],[h,h],'-',color='k',lw=1.5)
        axarr[0].plot([pp[0],pp[1]],[h,h],'-',color=col[c],lw=1.4,zorder=1000)
        hm.append(h)
        pos[c] = (pp[0]+pp[1])/2,h

    axarr[0].set_yticks(np.arange(0,max(hm)+(5*max(hm)/100), size_tick))
    axarr[0].set_ylim([0,max(hm)+(5*max(hm)/100)])
    axarr[0].grid()

    # ETICHETTE ASSE X
    step = 1  # mostra tutte le 50 label
    axarr[0].set_xticks(np.arange(0, len(labels_ord), step))
    axarr[0].set_xticklabels([labels_ord[i] for i in range(0, len(labels_ord), step)],
                             rotation=90, fontsize=fontsize_labels, ha='right')

    axarr[0].yaxis.tick_right()
    axarr[0].set_ylabel('Distance', fontsize=18)
    axarr[0].yaxis.set_label_position("right")

    if not nan:
        R = np.corrcoef(x)
    else:
        R =  np.array(pd.DataFrame(x.T).corr())

    ims = axarr[1].imshow(R, vmin=-1, vmax=1, cmap=cm.RdBu_r, interpolation='nearest')
    cbar_ax = fig.add_axes([0.95, 0.133, 0.03, 0.35])
    cbr = fig.colorbar(ims, cax=cbar_ax)
    cbr.set_label('correlation', fontsize=18)
    axarr[1].set_xlabel(xylabel, fontsize=18)
    axarr[1].set_ylabel(xylabel, fontsize=18)
    axarr[1].set_ylim([-0.5,N-0.5])

    for l in validated:
        if l == tuple(range(N)): continue
        axarr[1].add_patch(
            patches.Rectangle(
                (min(l)-0.5, min(l)-0.5),
                len(l),
                len(l),
                fill=False,
                lw=1.,
                color='green',
            )
        )

    plt.setp(axarr[0].get_xticklabels(), visible=True)
    plt.subplots_adjust(hspace= 0.5, wspace=0.1)
    axarr[0].set_title(f'{image_type} metric: {star}', fontsize=title_size)
    return fig