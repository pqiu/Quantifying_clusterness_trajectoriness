import numpy as np
import random
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import sklearn


def make_blob(num, mu_x, mu_y, sigma):
    #mu_, sigma = 0, 0.1 # mean and standard deviation
    X1 = np.random.normal(mu_x, sigma, num)
    Y1 = np.random.normal(mu_y, sigma, num)
    Blob = np.array([X1, Y1]).T
    return Blob

def make_blob_elipse(num, mu_x, mu_y, sigma_x, sigma_y):
    #mu_, sigma = 0, 0.1 # mean and standard deviation
    X1 = np.random.normal(mu_x, sigma_x, num)
    Y1 = np.random.normal(mu_y, sigma_y, num)
    Blob = np.array([X1, Y1]).T
    #plt.scatter(X1,Y1)
    return Blob

def gen_cluster(num=1000,seed=1):
    np.random.seed(seed)
    random.seed(seed)
    num_clusters = np.int(np.floor(np.random.exponential(scale=1.5)+2))
    prop = np.random.normal(5, 0.1, num_clusters) #SIGMA DETERMINES HOW UNBALANCED CLUSTERS ARE
    prop = prop/np.sum(prop)
    blobs = []
    center_cand_x = [int(i) for i in range(num_clusters)]
    center_cand_y = [int(i) for i in range(num_clusters)]
    for i in range(num_clusters):
        x=random.sample(center_cand_x,1)[0]
        y=random.sample(center_cand_y,1)[0]
        #print(x,y)
        blob1 = make_blob(int(num*prop[i]),x,y,np.random.normal(0.2,0.01))
        center_cand_x.remove(x)
        center_cand_y.remove(y)
        #blob1 = make_blob(int(num*prop[i]),random.randint(0,5),random.randint(0,5),np.random.normal(0.2,0.01))
        blobs.append(blob1)
    C = np.concatenate(blobs)
    return C

def gen_cluster_random(num=1000,seed= 1):
    np.random.seed(seed)
    random.seed(seed)
    num_clusters = np.random.randint(2,7)
    prop = np.random.uniform(0, 2, num_clusters) #SIGMA DETERMINES HOW UNBALANCED CLUSTERS ARE
    prop = prop/np.sum(prop)
    blobs = []
    center_cand_x = [int(i) for i in range(num_clusters)]
    center_cand_y = [int(i) for i in range(num_clusters)]
    for i in range(num_clusters):
        x=random.sample(center_cand_x,1)[0]
        y=random.sample(center_cand_y,1)[0]
        blob1 = make_blob_elipse(int(num*prop[i]),x,y,np.random.normal(0.3,0.02),np.random.normal(0.3,0.02))
        center_cand_x.remove(x)
        center_cand_y.remove(y)
        blobs.append(blob1)
    C = np.concatenate(blobs)
    return C

def gen_cluster_random_extreme(num=1000):
    num_clusters = np.random.randint(2,7)
    prop = np.random.uniform(0, 2, num_clusters) #SIGMA DETERMINES HOW UNBALANCED CLUSTERS ARE
    prop = prop/np.sum(prop)
    blobs = []
    center_cand_x = [int(i) for i in range(num_clusters)]
    center_cand_y = [int(i) for i in range(num_clusters)]

    xy = []
    for i in range(num_clusters):
        x=random.sample(center_cand_x,1)[0]
        y=random.sample(center_cand_y,1)[0]
        center_cand_x.remove(x)
        center_cand_y.remove(y)
        xy.append([x,y])
    width = np.sqrt(np.max(sklearn.metrics.pairwise_distances(np.array(xy))))
    for i in range(len(xy)):
        x=xy[i][0]
        y=xy[i][1]
        blob1 = make_blob_elipse(int(num*prop[i]),x,y,np.random.normal(width,width/10),np.random.normal(width,width/10))
        
        blobs.append(blob1)
    C = np.concatenate(blobs)
    return C

def gen_trajectory(num=1000,seed=1):
    np.random.seed(seed)
    random.seed(seed)
    if np.random.uniform() < 0.5:
        X2 = np.random.uniform(0, np.random.uniform(0.5,3), num)
    else:
        center=np.random.uniform(3,10)
        X2 = np.random.normal(center, 0.1*center, num)
    Y2 = np.sin(X2) + np.random.normal(0.15, np.random.uniform(0.01,0.05), num)
    C = np.array([X2,Y2]).T
    return C

def gen_trajectory_random(num=1000,seed=1):
    np.random.seed(seed)
    random.seed(seed)
    if np.random.uniform() < 0.5:
        X2 = np.random.uniform(0, np.random.uniform(0.5,5), num)
        Y2 = np.sin(X2) + np.random.normal(0.5, np.random.uniform(0.05,0.5), num)
        C = np.array([X2,Y2]).T
        return C
    else:
        width = np.random.normal(0.25,0.05)
        if width < 0:
            width = 0.2
        down = np.random.randint(1,2.5)
        up = np.random.randint(2.5,5)
        X2 = np.linspace(0, down, num=int(num/3))
        Y2 = X2 + np.random.normal(0, width, int(num/3))
        X3 = np.linspace(down, up, num=int(num/3))
        slope = np.random.uniform(-5,5)
        Y3 = slope*X2 + np.random.normal(0, width, int(num/3)) + Y2[-1]
        X4 = np.linspace(down, up, num=int(num/3))
        Y4 = (np.random.uniform(-5,5))*X2 + np.random.normal(0, width, int(num/3)) + Y2[-1]
        C = np.concatenate([np.array([X2,Y2]).T,np.array([X3,Y3]).T,np.array([X4,Y4]).T])
        return C

def gen_trajectory_random_extreme(num=1000):
    X2 = np.random.uniform(0, np.random.uniform(0.5,5), num)
    width = max(X2)-min(X2)
    Y2 = np.sin(X2) + np.random.normal(width, width, num)
    C = np.array([X2,Y2]).T
    return C


def gen_random(num=1000, mode='random',seed = 1):
    np.random.seed(seed)
    random.seed(seed)
    if mode == 'random':
        models = [gen_cluster_random,gen_trajectory_random]
        model_chosen = random.sample(models,1)[0]
    elif mode == 'trajectory':
        model_chosen = gen_trajectory_random
    elif mode == 'cluster':
        model_chosen = gen_cluster_random

    param = {}
    param['num'] = num
    param['seed'] = seed
    C = model_chosen(**param)
    if model_chosen == gen_cluster_random:
        ind = 0
    elif model_chosen == gen_trajectory_random:
        ind = 1
    return C, ind

def gen_random_extreme(num=1000, mode='random'):
    if mode == 'random':
        models = [gen_cluster_random_extreme,gen_trajectory_random_extreme]
        model_chosen = random.sample(models,1)[0]
    elif mode == 'trajectory':
        model_chosen = gen_trajectory_random_extreme
    elif mode == 'cluster':
        model_chosen = gen_cluster_random_extreme

    param = {}
    param['num'] = num
    C = model_chosen(**param)
    if model_chosen == gen_cluster_random_extreme:
        ind = 0
    elif model_chosen == gen_trajectory_random_extreme:
        ind = 1
    return C, ind

def gen_uniform(num=1000):
    lim = np.random.randint(2,8)
    X1 = np.random.uniform(0, lim, num)
    Y1 = np.random.uniform(0, lim, num)
    C = np.array([X1,Y1]).T
    return C

def plotImage(x, y, im):
    bb = Bbox.from_bounds(x,y,1,1)  
    bb2 = TransformedBbox(bb,ax.transData)
    bbox_image = BboxImage(bb2,
                        norm = None,
                        origin=None,
                        clip_on=False)

    bbox_image.set_data(im)
    ax.add_artist(bbox_image)