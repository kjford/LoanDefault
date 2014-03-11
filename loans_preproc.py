'''
process data from loan default Kaggle challenge 
'''
import pandas as pd
import numpy as np
#import cPickle
import pandamunger as pdm
#import NNtools as nnt


# defaults:
trfile='train_v2.csv'
testfile='test_v2.csv'
trdest='train_full'
testdest='test_full'
dadest='da.cp'

rowcropthresh=0.2 # crop any rows with more than rowcropthresh fraction NA values
#ndims=100
#epochs=15

'''
Munge Training data
'''

print('Loading data...')
# load training data
trdata=pd.read_csv(trfile)

# show some basic stats
nrows,ncol=trdata.shape
print('Raw training data has %d rows and %d columns'%(nrows,ncol))

# get rid of rows with sparse data
rthresh=int((1.0-rowcropthresh)*ncol)
trdata=trdata.dropna(thresh=rthresh)
nrows,ncol=trdata.shape
print('Culled training data has %d rows'%nrows)
# get id's and default values
trids=np.array(trdata.id)
trlabels=np.array(trdata.loss)
trdata=trdata.iloc[:,1:-1] # get rid of these columns

# cull some columns that are really long numbers that I am not sure how to interpret
longstr=(trdata.dtypes=='O')
# note: one other is a str in the test set, so get rid of it
longstr['f531']=True
trdata=trdata[trdata.columns[~np.array(longstr)]]
trdata,unicol=pdm.rmuni(trdata)
nrows,ncol=trdata.shape
print('Culled training data has %d columns'%ncol)

print('Fraction defaults: %f'%(trlabels>0).mean())
print('Average loss: %f'%(trlabels[trlabels>0].mean()))

print('Splitting categorical columns...')
trdata,newcols = pdm.onehot(trdata)
nrows,ncol=trdata.shape
print('Final training data has %d rows and %d columns'%(nrows,ncol))

print('Normalizing data...')
normvals = pdm.normcols(trdata)
# fill in other NA values
# using nearest neighbors takes way too long
print('Filling NA values...')
trdata.fillna(trdata.mean(),inplace=1)

print('Randomize')
np.random.seed(31415)
rp=np.random.permutation(nrows)
trdata=trdata.iloc[rp,:]
trlabels=trlabels[rp]
trids=trids[rp]

collist=trdata.columns.tolist()

'''
Test set
'''
print('Loading data from test set...')
testdata=pd.read_csv(testfile)
print('Done.')
testid=np.array(testdata.id)
testdata=testdata.iloc[:,1:]
# get rid of long strings
testdata=testdata[testdata.columns[~np.array(longstr)]]
testdata=pdm.rmuni(testdata,unicol)
# one hot:
print('Splitting categorical columns...')
testdata,newcols = pdm.onehot(testdata,mask=newcols)
# make sure this is in the correct order
testdata=testdata[collist]
print('Normalizing data...')
pdm.normcols(testdata, normvals=normvals)
print('Filling NA values')
testdata.fillna(trdata.mean(),inplace=1)

'''
print('Removing columns that are very different between train and test')
# this kills some columns that are useful...
coldiff=np.abs(trdata.mean()-testdata.mean())
colnew=trdata.columns[coldiff<5].tolist()
'''
colnew=trdata.columns.tolist()
trdata=trdata[colnew]
testdata=testdata[colnew]
nrowstest,ncol=trdata.shape
print('Final Data has %d columns'%ncol)


np.save(trdest,np.array(trdata))
np.save(testdest,np.array(testdata))
np.save('trainlabels',np.array(trlabels))
np.save('testids',np.array(testid))
np.save('collabels',np.array(colnew))
