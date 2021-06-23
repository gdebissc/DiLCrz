import sys, os
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale,normalize
from sklearn.decomposition import PCA

# VARIABLES
input_path = "to_treat" # folder with input csv files "index reactivity1 reactivity2 ..."
separator = '\t'
output_path = "clustered"
newline = '\n'
N = 4 # Number of clusters
NA_val = float(-1) # NA value format in input file
display = None # Whether to run normally ("None"), to run the "elbow" method or to draw a "PCA" plot


def GetListFile(PathFile):
    return [f for f in listdir(input_path) if isfile(join(input_path, f))]

def parseFile(files):
    f = open(os.path.join(input_path, files), 'r')
    lines = f.readlines()
    f.close()
    reac = []
    for line in lines:
        reac.append([float(x) for x in line.replace(',', '.').split(separator)])
    N_feat = len(reac[0])
    return reac,N_feat

def filter_nt(nucl):
    if len(list(filter(lambda x:x <= 0, nucl)))>0:
        return True
    else:
        return False

def standardize(X):
    X = np.log2(X).tolist()
    axes = [1,0] # 0 = col/temperature-wise / 1 = row/nucleotide-wise
    for a in axes:
        # Center & scale
        X = scale(
            X,
            with_mean = True, # center by mean
            with_std = False, # do not normalize std
            axis = a
        )
        # Normalize
        X = normalize(
            X,
            axis = a # 0 = normalize by columns / 1 = normalize by rows
        )
    return X

def makeFile(reac, clusters, files, N_feat):
    f = open(os.path.join(output_path, str('clustered_') + files), 'w')
    idx = 0
    for val in reac:
        if filter_nt(val) == True:
            f.write("-1\t" * (N_feat+1) + newline)
        else:
            f.write("\t".join([str(v).replace('.',',') for v in val])+"\t"+str(clusters[idx])+newline)
            idx = idx + 1
    f.close()
    print("  Created file clustered_{} with input data and cluster labels".format(files))


def elbowdisp(X):
    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()


def PCAdisp(X, N):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    km = KMeans(n_clusters = N).fit(X)
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], alpha=.5, c=km.labels_)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()

if __name__ == "__main__":
        
        for files in GetListFile(input_path):
            print('Treating file {}...'.format(files))
            reac = []
            reac, N_feat = parseFile(files)
            print("  {} features found".format(N_feat))
            X = reac[:]
            for nucl in reac:
                if filter_nt(nucl) == True:
                    X.remove(nucl)
            print("  {} out of {} nucleotides contain negative/ND values and are thus removed from the dataset".format(len(reac)-len(X),len(reac)))
            X = standardize(X)
            if display == "elbow":
                elbowdisp(X)
            elif display == "PCA":
                PCAdisp(X, N)
            else:
                km = KMeans(n_clusters = N).fit(X)
                makeFile(reac, km.labels_, files, N_feat)
                print("  Successfuly clustered with {} clusters".format(N))
            print("File {} treated.".format(files))
