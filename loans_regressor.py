'''
Fit regressor
'''

import numpy as np
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
import cPickle
from sklearn.ensemble import GradientBoostingRegressor

'''
load in munged training data
'''
trfile='train_full.npy'
foldcv=5

print('Loading up data...')
trdata=np.load(open(trfile,'rb'))
labels=np.load(open('trainlabels.npy','rb'))

classfeats=cPickle.load(open('bestfeatures.cp','rb'))
classmodel=cPickle.load(open('classmodel.cp','rb'))
classthr=cPickle.load(open('bestf1.cp','rb'))

tr_forclf=trdata[:,classfeats]
cv= cross_validation.StratifiedKFold(labels>0, n_folds=foldcv)
regmodel=GradientBoostingRegressor(max_depth=5,loss='lad')
maescore=list()
# fit using full data
for i,(tr,te) in enumerate(cv):
    # get class labels:
    trclass = classmodel.predict_proba(tr_forclf[tr])[:,1]>=classthr
    # fit regression
    trlabels=labels[tr]
    telabels=labels[te]
    regmodel.fit(trdata[tr[trclass],:],trlabels[trclass])
    # predict
    teclass=classmodel.predict_proba(tr_forclf[te])[:,1]>=classthr
    predvals=regmodel.predict(trdata[te[teclass],:])
    pred=np.zeros_like(teclass,dtype='float64')
    pred[teclass]=predvals
    # score mae
    mae=mean_absolute_error(telabels,pred)
    print(mae)
    maescore.append(mae)

print('Mean MAE on full set is: %4f'%np.mean(maescore))

# fit to whole dataset (typically GBM's do not overfit)
print('Fitting to all training data...')
regmodel.fit(trdata[labels>0],labels[labels>0])
print('Done')
cPickle.dump(regmodel,open('regmodel.cp','wb'))