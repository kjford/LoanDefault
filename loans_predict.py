import numpy as np
import cPickle
import csv


testfile='test_full.npy'
predfile='reducedlogisticreg_gbmreg.csv'
print('Loading up data...')
testdata=np.load(open(testfile,'rb'))
testid=np.load(open('testids.npy','rb'))

classfeats=cPickle.load(open('bestfeatures.cp','rb'))
classmodel=cPickle.load(open('classmodel.cp','rb'))
classthr=cPickle.load(open('bestf1.cp','rb'))
regmodel=cPickle.load(open('regmodel.cp','rb'))

print('Predicting class...')
forclf=testdata[:,classfeats]
testclass = classmodel.predict_proba(forclf)[:,1]>=classthr
print('Fraction positive class: %3f'%testclass.mean())
print('Predicting loss...')
predvals=regmodel.predict(testdata[testclass])
print('Average loss: %3f'%predvals.mean())
pred=np.zeros_like(testclass,dtype='float64')
pred[testclass]=predvals
print('Writing out')
header=['id','loss']

f = open(predfile,'wb')
csvf=csv.writer(f)
csvf.writerow(header)
csvf.writerows(zip(testid,pred))
f.close()