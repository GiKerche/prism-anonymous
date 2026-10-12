"""PCA/ridge computations from the frozen map-fitting implementation."""
import numpy as np

def require(ok,message):
    if not ok:raise ValueError(message)

def pca_train(matrix,maximum_rank):
    train=matrix[:1024];mean=train.mean(axis=0);centered=train-mean
    gram=centered@centered.T;values,left=np.linalg.eigh((gram+gram.T)/2)
    values=values[::-1];left=left[:,::-1]
    floor=max(float(values[0]),0.)*max(matrix.shape)*np.finfo(np.float64).eps*10
    numerical_rank=int((values>floor).sum())
    require(numerical_rank>=maximum_rank,'Insufficient benign PCA rank')
    basis=(centered.T@left[:,:maximum_rank])/np.sqrt(values[:maximum_rank])
    require(np.max(np.abs(basis.T@basis-np.eye(maximum_rank)))<=2e-6,'PCA basis not orthogonal')
    return dict(mean=mean,basis=basis,train=centered@basis,numerical_rank=numerical_rank,
                effective_rank=float(values[values>floor].sum()**2/(values[values>floor]**2).sum()))

def projection(matrix,indices,pca):
    centered=matrix[indices]-pca['mean']
    return centered@pca['basis'],np.sum(centered**2,axis=1)

def error(x,y,norm,weights):
    prediction=x@weights
    residual=np.maximum(norm+np.sum(prediction**2,axis=1)-2*np.sum(prediction*y,axis=1),0)
    require(norm.sum()>0,'Zero receiver variance')
    return float(residual.sum()/norm.sum())
