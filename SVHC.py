import meboot
from multiprocessing import cpu_count
import itertools
from functools import partial
import scipy.stats as st
from joblib import Parallel, delayed
import pandas as pd
import fastcluster
import numpy as np
import scipy as sp
import os
from numba import njit, prange
from tqdm import tqdm
import streamlit
os.environ["OMP_NUM_THREADS"] = "1"



def pre_cor(dati_5, tau_m):
    Z = np.zeros((dati_5.shape[0], dati_5.shape[1]))
    for i in range(Z.shape[0]):
        Z[i, :] = (dati_5[i, :] - np.mean(dati_5[i, :]))/np.std(dati_5[i, :])

    autocor = np.zeros((dati_5.shape[0], tau_m+1))
    for i in range(dati_5.shape[0]):
        for j in range(tau_m+1):
            serie = pd.Series(dati_5[i,:])
            autocor[i,j] = serie.autocorr(j)
    return Z, autocor




@njit(parallel=True)
def new_cor(dati_5, Z, autocor, tau_m):
    R = np.zeros((dati_5.shape[0], dati_5.shape[0]))

    for i in prange(R.shape[0]):
        Z_i = Z[i, :]
        for j in range(R.shape[0]):
            Z_j = Z[j, :]
            somma = 0
            for t1 in range(len(Z_i)):
                lim1 = max(t1 - tau_m, 0)
                lim2 = min(t1 + tau_m, Z_i.shape[0])
                for t2 in range(lim1, lim2, 1):
                    temp = np.abs(t1 - t2)
                    somma += Z_i[t1] * Z_j[t2] * autocor[i, temp]
            A = np.sum(autocor[i, :])
            R[i, j] = somma / (len(Z_i) * A)

    rho = np.zeros((R.shape[0], R.shape[1]))
    for i in prange(R.shape[0]):
        for j in range(R.shape[1]):
            rho[i, j] = R[i, j] / (R[i, i] * R[j, j])**(1/2)

    return rho


def inner_product(images_data):

    matrix_norm=np.zeros((len(images_data), len(images_data)))
    
    for i in range(len(images_data)):
        A=images_data[i]
        for j in range(len(images_data)):

            B=images_data[j]

            inner_product = np.dot(A, B)

            norm_A = np.linalg.norm(A)
            norm_B = np.linalg.norm(B)

            normalized_inner_product = inner_product / (norm_A * norm_B)

            if i==j:
                normalized_inner_product=int(normalized_inner_product)

            matrix_norm[i][j]=np.sqrt(2*(1-normalized_inner_product))
            
    return matrix_norm


def distance_fisher(P, Q):
    
    kl_div_PQ = np.sum(sp.special.rel_entr(P, Q))
    kl_div_QP = np.sum(sp.special.rel_entr(Q, P))
    
    sym_kl= kl_div_PQ +  kl_div_QP
    
    HP= -np.sum(P * np.log(P))
    HQ = -np.sum(Q * np.log(Q))
    
    norm_sym_kl=sym_kl/(np.sqrt(HP*HQ))
    
    return norm_sym_kl

def kullback_leibler(images_data):
    
    matrix_norm=np.zeros((len(images_data), len(images_data)))

    for i in range(len(images_data)):

        A=images_data[i]
        for j in range(len(images_data)):

            B=images_data[j]
            matrix_norm[i][j]=distance_fisher(A, B)
            
    return matrix_norm

            
def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def flatx(x):
    return tuple(flatten(x))


def dist(R, method):
    N = R.shape[0]
    d = R[np.triu_indices(N, 1)]

    if method == 'average':
        out = fastcluster.average(d)
    if method == 'complete':
        out = fastcluster.complete(d)
    if method == 'single':
        out = fastcluster.single(d)

    outI = out.astype(int)
    dend = {i: (i,) for i in range(N)}

    for i in range(len(outI)):
        dend[i + N] = (dend[outI[i][0]], dend[outI[i][1]])

    for i in range(N):
        dend.pop(i, None)

    dend = [(flatx(a), flatx(b)) for a, b in dend.values()]

    dend = {flatx((a, b)): (np.array(a), np.array(b)) for a, b in dend}

    return dend


def singPV(Rb, method):

    if method == 'average':
        rxy = dict(
            map(lambda x: (x[0], Rb[x[1][0]][:, x[1][1]].mean()), LV.items()))
    if method == 'complete':
        rxy = dict(
            map(lambda x: (x[0], Rb[x[1][0]][:, x[1][1]].max()), LV.items()))
    if method == 'single':
        rxy = dict(
            map(lambda x: (x[0], Rb[x[1][0]][:, x[1][1]].min()), LV.items()))

    PV = list(map(lambda x: int(rxy[x[1]] < rxy[x[0]]), gen.items()))

    return PV


def SingleBoot(nan, method, metric, bootstrap, tau_m, sel):

    if nan == False and bootstrap == 'simple':
        Xb = Xg[:, sel]
    elif nan == False and bootstrap == 'meboot':
        Xb = meb.bootstrap()
    #print(Xb.shape, Xb)
    if not nan:
        if metric == 'pearson':
            Rb = 1. - np.corrcoef(Xb)
        elif metric == 'spearman':
            Rb = 1. - st.spearmanr(Xb.T)[0]
        elif metric == "new_cor":
            Z, autocor = pre_cor(Xb, tau_m)
            rho = new_cor(Xb, Z, autocor, tau_m)
            Rb = 1. - rho
        elif metric == 'inner_product':
            Rb = inner_product(Xb)
        elif metric == 'kullback_leibler':
            Rb = kullback_leibler(Xb)
    else:
        Rb = 1. - Xg[sel].T.corr(method=metric).values

    return singPV(Rb, method)


def BootDist(
        method,
        Nt=1000,
        nan=False,
        ncpu=1,
        metric='pearson',
        bootstrap='simple',
        tau_m = 50):

    global gen
    gen = {tuple(s): k for k in LV for s in LV[k] if len(s) > 1}

    if bootstrap == 'simple':
        sels = np.random.choice(
            range(
                Xg.shape[1]), replace=True, size=(
                Nt, Xg.shape[1]))
    elif bootstrap == 'meboot':
        sels = range(Nt)
        global meb
        meb = meboot.MEBOOT(Xg)

    f = partial(SingleBoot, nan, method, metric, bootstrap, tau_m)
    #print(f)
    PV = Parallel(
        n_jobs=ncpu,
        backend="threading")(
        delayed(f)(sel) for sel in tqdm(sels))

    return sorted(zip(np.array(PV).mean(axis=0), gen.keys()))


def FDR(PV, alpha, N):
    p = np.array(list(zip(*PV))[0])
    thr = np.arange(1, len(PV) + 1) * alpha / len(PV)

    sel = np.where(p < thr)[0]
    if len(sel) == 0:
        thr = -1.
    else:
        thr = thr[sel][-1]

    L = [range(N)] + [PV[i][1] for i in np.where(p <= thr)[0]]
    L = list(map(tuple, sorted(L, key=len, reverse=True)))

    PV = {c: p for p, c in PV}
    PV[tuple(range(N))] = np.nan
    return L, PV

def Analytic_PV(r,n):
    R = 1-r
    gen = {tuple(s):k for k in LV for s in LV[k] if len(s)>1}

    sd = np.vectorize(lambda x: (1./n)*(1-x**2)**2)
    COV = lambda i,j: (0.5/n)*(((R[i[0],j[0]]-R[i[0],i[1]]*R[j[0],i[1]])*(R[i[1],j[1]]-R[i[1],j[0]]*R[j[0],j[1]]))+((R[i[0],j[1]]-R[i[0],j[0]]*R[j[0],j[1]])*(R[i[1],j[0]]-R[i[1],i[0]]*R[i[0],j[0]]))+((R[i[0],j[0]]-R[i[0],j[1]]*R[j[1],j[0]])*(R[i[1],j[1]]-R[i[1],i[0]]*R[i[0],j[1]]))+((R[i[0],j[1]]-R[i[0],i[1]]*R[i[1],j[1]])*(R[i[1],j[0]]-R[i[1],j[1]]*R[j[1],j[0]])))
    PV = []
    for h,(son,fat) in enumerate(gen.items()):


        a_s,b_s = LV[son]
        a_f,b_f = LV[fat]

        m = R[a_s][:,b_s].mean()-R[a_f][:,b_f].mean()

        ss = sd(R[a_s][:,b_s])
        sf = sd(R[a_f][:,b_f])

        elm_f = list(itertools.product(a_f, b_f))
        elm_s = list(itertools.product(a_s, b_s))

        ns = 1./np.size(ss)
        nf = 1./np.size(sf)

        cf = (nf**2)*2*sum(COV(elm_f[i],elm_f[j]) for i in range(len(elm_f)) for j in range(i))
        cs = (ns**2)*2*sum(COV(elm_s[i],elm_s[j]) for i in range(len(elm_s)) for j in range(i))
        cmix = -2*(ns*nf)*sum(COV(elm_f[i],elm_s[j]) for i in range(len(elm_f)) for j in range(len(elm_s)))

        x = (ns**2)*ss.sum()+(nf**2)*sf.sum()+cf+cs+cmix

        s = np.sqrt(x)

        PV.append((st.norm.cdf(0,m,s),son))

    return PV


def Find_ValidatedCluster(
        X,
        Nt=1000,
        alpha=0.05,
        ncpu=cpu_count(),
        dendrogram=None,
        method='average',
        metric='pearson',
        bootstrap='simple',
        tau_m = 50):

    global sd
    sd = np.random.RandomState(0)

    nan = np.isnan(X).any()
    if nan and bootstrap == 'meboot':
        raise Exception('meeboot does not accept nan')

    if not nan:
        if metric == 'pearson':
            R = 1. - np.corrcoef(X)
        elif metric == 'spearman':
            R = 1. - st.spearmanr(X.T)[0]
        elif metric == "new_cor":
            Z, autocor = pre_cor(X, tau_m)
            rho = new_cor(X, Z, autocor, tau_m)
            R= 1. - rho
        elif metric == 'inner_product':
            R = inner_product(X)
        elif metric == 'kullback_leibler':
            R = kullback_leibler(X)
            
    else:
        X = pd.DataFrame(X)
        R = 1. - X.T.corr(method=metric).values

    global Xg
    Xg = X

    global LV
    if dendrogram is None:
        LV = dist(R, method)
    else:
        LV = dendrogram

    if Nt == 0:
        PV = Analytic_PV(R, X.shape[1])
    else:
        PV = BootDist(method, Nt, nan, ncpu, metric, bootstrap, tau_m)

    PV = [(p, tuple(sorted(k))) for p, k in PV]

    L, PV = FDR(PV, alpha, X.shape[0])

    return {'validated': L, 'pvalues': PV, 'dendrogram': LV}





def clusterize(sn, image_type, images, bands):
    from plot_utils import DendroAndCorrDists 
    bands = np.array(bands)
    clust =Find_ValidatedCluster(
        images,
        method='average',
        metric="inner_product",
        tau_m=50
    )
    
    fig = DendroAndCorrDists(
        images,
        sn,
        image_type,
        bands,
        clust['validated'],
        clust['dendrogram'],
        method='average',
        metric="inner_product",
        tau_m=50,
        size_tick=0.05)
    streamlit.pyplot(fig)
    return fig