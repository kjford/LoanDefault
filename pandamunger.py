'''
data munging routines for pandas
'''
import numpy as np
import pandas as pd

def rmuni(DF, rmcol=None):
    '''
    Remove univariate column
    or columns in rmcol
    '''
    DFout=DF.copy()
    if rmcol:
        for k in rmcol:
            DFout=DFout.drop(k,axis=1)
        return DFout
    else:
        rmcol=[]
        for k in DF:
            data=DF[k][~DF[k].isnull()]
            uinds=data.unique()
            if len(uinds)<2:
                DFout=DFout.drop(k,axis=1)
                rmcol.append(k)
        return (DFout,rmcol)
            

def onehot(DF, mask=None, maxbins=10):
    '''
    usage:
    DFout,mask = onehot(DF , mask=None, maxbins=10)
    split columns of pandas data frame containing categorical data into binary columns
    optional inputs:
    mask (None):
    A dictionary with keys corresponding to columns in input data frame
    and values containing the values of each bin
    If none given, then will return a mask created by the binning process
    maxbins (10):
    Maximum number of unique values in a column to be considered categorical data
    output:
    DFout:
    Dataframe DF with rows same as input and new columns
    containing one-hot split columns named 'originalcolumnname_binvalue'
    mask:
    If no mask is given as input, then a dictionary containing columns in input data frame
    and values containing the values of each bin
    nan values are assigned to each bin if row has nan or value not in the mask
    '''
    DFout=DF.copy()
    if mask:
        for k in mask:
            data=DF[k]
            uinds=mask[k]
            for u in uinds:
                newkeyname = '%s_%s' %(k,u)
                datamask = 1.0*(data==u)
                datamask=np.float64(datamask)
                if sum(np.isnan(data))>0:
                        datamask[np.isnan(data)]=np.nan
                DFout[newkeyname]=datamask
            
    else:
        mask={}
        for k in DF:
            data=DF[k][~DF[k].isnull()]
            uinds=data.unique()
            if len(uinds)<=maxbins and len(uinds)>2 and data.dtype!='float64':
                mask[k]=uinds
                for u in uinds:
                    newkeyname = '%s_%s' %(k,u)
                    datamask = 1.0*(data==u)
                    datamask=np.float64(datamask)
                    if sum(np.isnan(data))>0:
                        datamask[np.isnan(data)]=np.nan
                    DFout[newkeyname]=datamask
                
    for dk in mask:
        DFout=DFout.drop(dk,axis=1)
    return (DFout,mask)

def normcols(DF, bound=False, normvals=None, skewthresh=None):
    '''
    usage:
    normvals = normcols(DF, normvals=None, skewthresh=2)
    
    Normalize data in columns to [0 1] if bound
    or mean subtracted unit variace if not
    transforms data if not very normal looking
    works in-place
    returns normvals as a dict of column names and list
    with values of transformation ((0,0) for none, (1,offset) for log+offset)
    min and (max-min)
    '''
    if normvals:
        for k in normvals:
            if normvals[k][0][0]==1:
                offset=normvals[k][1]
                DF[k]=DF[k]+offset
                DF[k]=np.log(DF[k])
            newvals=(DF[k]-normvals[k][1])/normvals[k][2]
            newvals[~np.isfinite(newvals)]=DF[k][np.isfinite(newvals)].median()
            DF[k]=newvals
    else:
        normvals={}
        for k in DF:
            normvals[k]=[]
            data=DF[k][~DF[k].isnull()]
            uinds=data.unique()
            if len(uinds)<3: # leave binary columns alone, just zero
                m=DF[k].min()
                s=DF[k].max() - m
                normvals[k].append((0,0))
            else:
                if skewthresh:
                    skw=DF[k].skew()
                else:
                    skw=0
                if skw>skewthresh and skewthresh:
                    if DF[k].min()>0:
                        DF[k]=np.log(DF[k])
                        normvals[k].append((1,0))
                    else:
                        offset=DF[k].min()
                        if offset==0:
                            offset=1
                        else:
                            offset=-2*offset
                    
                        DF[k]=DF[k]+offset
                        DF[k]=np.log(DF[k])
                        normvals[k].append((1,offset))
                else:
                    normvals[k].append((0,0))
                if bound:
                    m=DF[k].min()
                    s=DF[k].max() - m
                else:
                    m=DF[k].mean()
                    s=DF[k].std() 
                if s==0:
                    s=1.0
            
            newvals=(DF[k]-m)/s
            
            DF[k]=newvals
            normvals[k].append(m)
            normvals[k].append(s)
        return normvals
        

def fillna_knn(DF, nhood=5):
    '''
    Fill nan values with values from nearest neighbors
    this scales with n^2 so not feasible for large datasets
    '''
    # find rows with missing values
    toclean=pd.isnull(DF).any(axis=1)
    for i in range(len(toclean)):
        if toclean[i]:
            # get nhood nearest neighbors
            nd=1/np.zeros(nhood) # hold distances
            ni=np.zeros(nhood,dtype=int) # hold indices
            ivals=np.array(DF.iloc[i,:])
            for j in range(len(toclean)):
                if i!=j:
                    dsquared=(ivals-np.array(DF.iloc[j,:]))**2
                    dsquared=np.nansum(dsquared)/np.sum(np.isfinite(dsquared))
                    if dsquared<nd[-1]:
                        nd=np.delete(nd,-1)
                        ni=np.delete(ni,-1)
                        nd=np.append(nd,dsquared)
                        ni=np.append(ni,j)
                        ni=ni[nd.argsort()]
                        nd.sort()
            # fill in values using closest neighbors
            count=0
            for v in ivals:
                if np.isnan(v):
                    closestvals = np.array(DF.iloc[ni,count])
                    usethis = np.median(closestvals[np.isfinite(closestvals)])
                    DF.iloc[i,count]=usethis
                count+=1

def getGroupStats(DF,statkey,lookupkey):
    '''
    Get average stats for column statkey for each group in lookupkey
    Assumes data that is binned in lookupkey
    returns a list with averages, counts, and subgroup name for each group
    '''
    statseries = np.array(DF[statkey])
    lookupseries = np.array(DF[lookupkey])
    uvals = np.unique(lookupseries)
    avgs = np.zeros(len(uvals))
    counts = np.zeros(len(uvals))
    for i in range(len(uvals)):
        subgroup = (statseries[lookupseries==uvals[i]])
        subgroup = subgroup[~np.isnan(subgroup)]
        avgs[i]= subgroup.mean()
        counts[i] = subgroup.size
        
    return list(zip(avgs,counts,uvals))

