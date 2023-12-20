import matplotlib.pyplot as plt
from anndata import AnnData
import scanpy_modified as scanpy
from tqdm import tqdm
import numpy as np
from numpy import inf
from ripser import Rips
from scipy.spatial import distance
from numpy import linalg as LA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import sklearn
import pandas as pd
from sklearn.manifold import TSNE
import umap
import glob, os
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import time
import sys
from scipy import sparse
from scipy.stats import rankdata
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sknetwork.clustering import Louvain
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import random

eps = sys.float_info.epsilon

def preprocessing(data):
    data = density_downsampling(data, od = 0.05, td = 1)
    data = normalize(data)
    return(data)

def preprocessing2(data):
    data = density_downsampling(data, od = 0.1, td = 1)
    data = normalize(data)
    return(data)

def preprocessing3(data):
    data = density_downsampling(data, od = 0.01, td = 1)
    data = normalize(data)
    return(data)

def preprocessing4(data):
    data = normalize(data)
    return(data)

def preprocessing5(data):
    data = density_downsampling(data, od = 0.05, td = 0.7)
    data = normalize(data)
    return(data)

def preprocessing6(data):
    data = density_downsampling(data, od = 0.05, td = 0.5)
    data = normalize(data)
    return(data)

def preprocessing7(data):
    data = density_downsampling(data, od = 0.05, td = 0.3)
    data = normalize(data)
    return(data)

# def preprocessing5(data):
#     if len(data) < 1000:
#         data = density_downsampling(data, od = 0.05, td = 1)
#     if len(data) > 1000 and len(data) < 3000:
#         data = density_downsampling(data, od = 0.05, td = 0.5)
#     if len(data) > 3000:
#         data = density_downsampling(data, od = 0.05, td = 0.1)
#     data = normalize(data)
#     return(data)

def normalize(data):
    normalized = (data - np.min(data, axis = 0)) / (np.max(data, axis = 0) - np.min(data, axis = 0))
    return normalized 

def density_downsampling(Data, od = 0.05, td = 1):
    dist = distance.pdist(Data, metric='euclidean')    
    dist_m = distance.squareform(dist)
    sorted_dist_m = np.sort(dist_m)
    median_min_dist = np.median(sorted_dist_m[:,1])
    #dist_thres = 5 * median_min_dist
    dist_thres = np.max(median_min_dist)

    local_densities = np.sum(1*(dist_m < dist_thres),0)
    #print(local_densities)
    OD = np.quantile(local_densities, od)
    TD = np.quantile(local_densities, td)

    seed_value = 42
    np.random.seed(seed_value)
    #print(OD,TD)
    IDX_TO_KEEP = []
    for i in range(len(local_densities)):
        if local_densities[i] < OD:
            continue
        elif local_densities[i] > TD:
            if np.random.uniform(0,1) < TD/local_densities[i]:
                 IDX_TO_KEEP.append(i)
        else:
            IDX_TO_KEEP.append(i)
    downsampled_data = Data[IDX_TO_KEEP,:]
    return downsampled_data


def features_dpt_entropy(data, num_bins = 10, visualize = False):
    num_data = len(data)
    data = AnnData(data)
    data.uns['iroot'] = 0
    scanpy.pp.neighbors(data,n_neighbors=max(10,int(0.005*num_data)), method='umap',knn=True)
    scanpy.tl.diffmap(data)
    scanpy.tl.dpt(data)
    tmp = np.stack(data.obs['dpt_distances'])
    tmp[tmp == inf] = 1.5 * np.max(tmp[tmp != inf]) 
    tmp[tmp == -inf] = -1 * np.min(tmp[tmp != -inf]) 
    a = plt.hist(tmp[np.triu(tmp, 1) != 0], bins = num_bins)
    hs = a[0]/np.sum(a[0])
  
    ent = entropy(hs, base=num_bins)
    if visualize == True:
        plt.show()
    elif visualize == False:
        plt.close()
    return ent


def features_homology_dpt_entropy(data, num_bins = 3, visualize = False):
    data = AnnData(data)
    data.uns['iroot'] = 0
    #scanpy.pp.neighbors(data,n_neighbors=5, method='umap', knn=True)
    scanpy.pp.neighbors(data)
    scanpy.tl.diffmap(data)
    scanpy.tl.dpt(data)
    tmp = np.stack(data.obs['dpt_distances'])
    tmp[tmp == inf] = np.random.normal(1.5, 0.1) * np.max(tmp[tmp != inf]) 
    tmp[tmp == -inf] = -1 * np.min(tmp[tmp != -inf]) 
    rips = Rips(maxdim=0,verbose = False)
    diagrams = rips.fit_transform(sparse.csr_matrix(tmp),distance_matrix = True)
    a = plt.hist(diagrams[0][:-1,1], bins = num_bins)
    hs = a[0]/np.sum(a[0])   
    ent = entropy(hs, base=num_bins)
    ent = np.log(ent)
    if visualize == True:
        plt.show()
    elif visualize == False:
        plt.close()
    return ent

def features_vector(data, cells_per_cluster = 20, metric = 'euclidean'):
    _, dim = data.shape
    if dim > 5:
        pca = PCA(n_components=5)
        data = pca.fit_transform(data)
    _, dim = data.shape
    X = data
    num_clusters = int(len(X)/int(cells_per_cluster))
    if num_clusters > 100:
        num_clusters = 100
    SCORES = 0
    num_rep = 10
    for rep in range(num_rep):
        np.random.seed(rep)
        kmeans = KMeans(n_clusters=num_clusters, random_state=rep).fit(X)
        for m in range(num_clusters):

            clusters = kmeans.cluster_centers_.tolist()
            all_dist = sklearn.metrics.pairwise_distances(clusters, metric=metric)
            threshold = np.percentile(all_dist[np.triu(all_dist, 1) !=0], 20)
            index = m
            dist = sklearn.metrics.pairwise_distances(np.array(clusters[index]).reshape(1, -1),clusters, metric=metric)[0]
            current_index = index
            kmean_order = []
            for i in range(len(clusters)):
                current_kmean = clusters[current_index]
                kmean_order.append(current_kmean)
                clusters.remove(clusters[current_index])
                dist_current = sklearn.metrics.pairwise_distances(np.array(current_kmean).reshape(1, -1), clusters, metric=metric)[0]
                if len(dist_current) == 1:
                    break
                next_index = np.argsort(dist_current)[0]  
                if dist_current[next_index] > threshold:
                    break
                current_index = next_index
            vectors = []
            for j in range(len(kmean_order)-1):
                d1 = np.array(kmean_order[j])
                d2 = np.array(kmean_order[j+1])
                vectors.append(d2-d1)
            a = np.sum(vectors,axis = 0)
            try:
                norm = LA.norm(np.sum(vectors,axis = 0),ord=dim)
                SCORES += norm
            except Exception:
                pass
    return SCORES/(num_clusters*num_rep)

def features_ripley_dpt_v2(data, thresholdSize=100, visualize = False):
    X = data
    n, dim = X.shape
    MIN_MAX = []
    for i in range(dim):
        dim_min = np.min(X[:,i])
        dim_max = np.max(X[:,i])
        MIN_MAX.append((dim_min,dim_max))
    num_repeats = 1
    rScore = 0 
    for i in range(num_repeats):
        SDATA = []
        for i in range(len(MIN_MAX)):
            shuffled_data = np.random.uniform(MIN_MAX[i][0],MIN_MAX[i][1],n)
            SDATA.append(shuffled_data)
           
        SX = np.array(SDATA).T

        data = AnnData(X)
        data.uns['iroot'] = 0
        scanpy.pp.neighbors(data)
        scanpy.tl.diffmap(data)
        scanpy.tl.dpt(data)
        DX = np.stack(data.obs['dpt_distances'])

        data = AnnData(SX)
        data.uns['iroot'] = 0
        scanpy.pp.neighbors(data)
        scanpy.tl.diffmap(data)
        scanpy.tl.dpt(data)
        DSX = np.stack(data.obs['dpt_distances'])        
        DX[DX == inf] = 1.5 * np.max(DX[DX != inf]) 
        DSX[DSX == inf] = 1.5 * np.max(DSX[DSX != inf]) 

        xs = np.linspace(0,np.nanmax(DX[DX != -np.inf])+1,thresholdSize)
        T_K_DX = []
        for i in range(len(xs)):
            th = xs[i]
            K_DX = np.sum( np.sum(1*(DX < th), axis = 0)/n) 
            T_K_DX.append(K_DX)
    
    
        xs = np.linspace(0,np.nanmax(DSX[DSX != -np.inf])+1,thresholdSize)
        T_K_DSX = []
        for i in range(len(xs)):
            th = xs[i]
            K_DSX = np.sum( np.sum(1*(DSX < th), axis = 0)/n) 
            T_K_DSX.append(K_DSX)

        dx_n = normalize(T_K_DX)
        dsx_n = normalize(T_K_DSX)
        sum_n = np.array(dx_n) + np.array(dsx_n)
        dx_n = dx_n[sum_n != 2]
        dsx_n = dsx_n[sum_n != 2]

        Score = np.trapz(np.abs(dx_n - dsx_n), dx = 1/len(dx_n))
        rScore += Score
    return rScore / num_repeats


def features_avg_connection_dpt(df):
    SCORE = []
    c = density_downsampling(df,od = 0.03, td = 0.3)
    K = np.linspace(0.03, 1, 20)    
    k_scores = []
    for k in K:
        sc = generate_score_k_dpt(c, k)
        k_scores.append(sc)
    score = np.trapz(k_scores, K/np.max(K))
    return score

def generate_score_k_dpt(tmp_data, k):
    np.random.seed(42)
    if len(tmp_data) > 200:
        num_repeats = 5
        final_score = 0
        for i in range(num_repeats):
            idx = np.random.randint(0,len(tmp_data), size=200)
            t_data = tmp_data[idx,:]

            data = AnnData(t_data)
            data.uns['iroot'] = 0
            scanpy.pp.neighbors(data)
            scanpy.tl.diffmap(data)
            scanpy.tl.dpt(data)
            DX = np.stack(data.obs['dpt_distances'])
            DX[DX == inf] = 1.5 * np.max(DX[DX != inf]) 
            
            K = int(len(t_data) * k)

            knn_distance_based = (
                NearestNeighbors(n_neighbors=int(len(t_data) * k), metric="precomputed")
                    .fit(DX)
            )
            
            A = knn_distance_based.kneighbors_graph(DX).toarray()
            
            SA = 1*((A + A.T) > 1.5)
            old_total = 0 
            for i in range(10):
                total = np.sum(SA)
                if total == old_total:
                    break
                else:
                    old_total = total
                SA = np.matmul(SA,SA)
                SA = 1*(SA > 1)

            avg_connect = np.sum(SA,axis=0) / len(SA)
            final_score += np.median(avg_connect)

        return final_score/num_repeats

    else:
        data = AnnData(tmp_data)
        data.uns['iroot'] = 0
        scanpy.pp.neighbors(data)
        scanpy.tl.diffmap(data)
        scanpy.tl.dpt(data)
        DX = np.stack(data.obs['dpt_distances'])
        DX[DX == inf] = 1.5 * np.max(DX[DX != inf]) 
        
        knn_distance_based = (
            NearestNeighbors(n_neighbors=max(1,int(len(tmp_data) * k)), metric="precomputed")
                .fit(DX)
        )
        
        A = knn_distance_based.kneighbors_graph(DX).toarray()
        
        SA = 1*((A + A.T) > 1.5)
        old_total = 0 
        for i in range(10):
            total = np.sum(SA)
            if total == old_total:
                break
            else:
                old_total = total
            SA = np.matmul(SA,SA)
            SA = 1*(SA > 1)
        avg_connect = np.sum(SA,axis=0) / len(SA)
        final_score = np.median(avg_connect)
    
        return final_score

scanpy.settings.verbosity = 0


def sc_vis(file):
    print('reding ... {}'.format(file))
    df = pd.read_csv(file, index_col=0)
    df = preprocessing(np.array(df))
    print('size of the file is {}'.format(df.shape))
    data = np.array(df)
    data = AnnData(data)
    data.uns['iroot'] = 0 
    plt.figure(figsize=(8,8))
    scanpy.set_figure_params(dpi=80, dpi_save=150, figsize=(5,5))
    scanpy.tl.pca(data, svd_solver='arpack')
    scanpy.pp.neighbors(data)
    scanpy.tl.diffmap(data)
    scanpy.tl.dpt(data)
    scanpy.pl.diffmap(data,color=['dpt_pseudotime'])
    
    pca = PCA(n_components=20)
    embedding_pca = pca.fit_transform(df)
    A = kneighbors_graph(embedding_pca, 10, mode='connectivity', include_self=True)
    louvain = Louvain()
    labels = louvain.fit_transform(A)
    Umap = umap.UMAP()
    embedding_umap = Umap.fit_transform(embedding_pca)
    fig = plt.figure(figsize=(5,5))
    plt.scatter(embedding_umap[:,0],embedding_umap[:,1],c=labels, alpha=1, cmap='tab20')
    plt.xticks([])
    plt.yticks([])
    plt.grid(b=None)
    plt.show()
    #plt.savefig('{}.png'.format(file))

def scoring(df,num_downsample = 5000):
    if len(df) > num_downsample:
        tmp = []
        for i in range(3):
            np.random.seed(i)
            random.seed(i)
            df = df[random.sample(range(len(df)), num_downsample),:]
            df = preprocessing(df)
            sc1 = features_dpt_entropy(df, num_bins = 10)
            sc2 = features_homology_dpt_entropy(df,num_bins = 3)
            sc3 = features_vector(df)
            sc4 = features_ripley_dpt_v2(df)
            sc5 = features_avg_connection_dpt(df)
            tmp.append([sc1,sc2,sc3,sc4,sc5])
        scores = list(np.median(np.stack(tmp),axis=0))
    else:
        df = preprocessing(df)
        sc1 = features_dpt_entropy(df, num_bins = 10)
        sc2 = features_homology_dpt_entropy(df,num_bins = 3)
        sc3 = features_vector(df)
        sc4 = features_ripley_dpt_v2(df)
        sc5 = features_avg_connection_dpt(df)
        scores = [sc1,sc2,sc3,sc4,sc5] 
    return scores

def explain_score(sc_traj_clstr_score):
    META_SCORES = list(np.load('data/simulated_metascores_12000.npy')) # Loading pre-computed scores for simulated datasets

    clstr = META_SCORES[:3000]
    traj = META_SCORES[3000:6000]
    clstr_r1 = META_SCORES[6000:9000]
    traj_r1 = META_SCORES[9000:12000]
   

    npy_sim = np.array(META_SCORES)
    feature_names = ['P-dist','Homology','Vector','Ripleys','Deg. of Sep.']

    metric = 'euclidean'
    seed = 1
    n_neighbors = 50
    min_dist = 0.6
    figsize = 5

    scaler = StandardScaler()
    tmp_np = scaler.fit_transform(npy_sim)
    tmp_reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2,random_state=seed,min_dist=min_dist, metric=metric)
    embedding = tmp_reducer.fit_transform(tmp_np)

    classes = ['Clear Clusters','Clear Trajectory','Noisy Trajectory','Noisy Clusters']
    c1 = [0 for i in range(3000)]
    c2 = [1 for i in range(3000)]
    c3 = [0 for i in range(3000)]
    c4 = [1 for i in range(3000)]


    c = c1+c2+c3+c4

    neigh = KNeighborsClassifier(n_neighbors=100)
    neigh.fit(embedding, c)

    b = neigh.predict(embedding)
    p = neigh.predict_proba(embedding)[:,1]

    ####################
    ####################
    INPUT_SCALED = scaler.transform(np.array(sc_traj_clstr_score).reshape(1, -1))
    UMAP_PROJECTION = tmp_reducer.transform(INPUT_SCALED.reshape(1, -1))
    PREDICTION = neigh.predict(UMAP_PROJECTION)
    ####################
    ####################

    plt.figure(figsize=(figsize,figsize))
    plt.scatter(embedding[:,0],embedding[:,1], c = p, alpha = 1)
    plt.scatter(UMAP_PROJECTION[0][0],UMAP_PROJECTION[0][1],color='red',marker='^')
    plt.show()

    for i in range(len(feature_names)):
        feat = i
        plt.figure(figsize=(5,3))
        plt.violinplot([np.array(clstr)[:,feat],
                        np.array(clstr_r1)[:,feat],
                        np.array(traj_r1)[:,feat],
                        np.array(traj)[:,feat]],
                      showmeans = True, showextrema=False)
        plt.axhline(sc_traj_clstr_score[i],c='red',ls='--')
        plt.title(feature_names[i], fontsize = 20)
        plt.xticks(fontsize=15, rotation=315)
        plt.xticks([1, 2, 3, 4], ['Clear Clusters','Noisy Clusters','Noise Trajectory','Clear Trajectory'])
        plt.show()
    