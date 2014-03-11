import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import auc_score, f1_score
import cPickle

trfile='train_full.npy'
#testfile='test_full.cp'
foldcv=5
#predfile='ensembleprediction.csv'

print('Loading up data...')
trdata=np.load(open(trfile,'rb'))
labels=np.load(open('trainlabels.npy','rb'))
collab=np.load('collabels.npy')

nrows,ncol=trdata.shape

labels2=labels>0

split=int(np.round(nrows/(1.0*foldcv)))
tr=range(nrows)[:split]
te=range(nrows)[split:]
'''
Logistic regression on reduced set
'''
# make models using very reduced features
# these 2 are good
f527col=np.where(collab=='f527')[0][0]
f528col=np.where(collab=='f528')[0][0]

tr_tran=[trdata[:,(f527col,f528col)]]
scores=np.zeros_like(collab)
lr=LogisticRegression(C=1e10) # logistic regression with weak l2 penalty
aucholder=np.zeros(ncol)
# add one more feature
for i in range(ncol):
    tr2=trdata[:,(f527col,f528col,i)]
    lr.fit(tr2[tr],labels2[tr])
    a=auc_score(labels2[te],lr.predict_proba(tr2[te])[:,1])
    print('added %s auc: %3f'%(collab[i],a))
    aucholder[i]=a
    
# successively add best features (first 20)
tr2=trdata[:,(f527col,f528col)]
sortedfeatinds = aucholder.argsort()[::-1] # best first
Cs=np.logspace(0,10,5)
bestc=np.zeros(20)
besta=np.zeros(20)
for i in range(20):
    toadd=sortedfeatinds[i]
    tr2=np.append(tr2,trdata[:,toadd].reshape(nrows,1),axis=1)
    print('adding feature: %s number: %d'%(collab[toadd],i))
    for c in Cs:
        lr.C=c
        lr.fit(tr2[tr],labels2[tr])
        a=auc_score(labels2[te],lr.predict_proba(tr2[te])[:,1])
        print('AUC: %3f with C: %f'%(a,c))
        if a>besta[i]:
            bestc[i]=c
            besta[i]=a

# best score of all:
bestoveralla=besta.argsort()[-1]
bestcforbesta=bestc[bestoveralla]
print('best auc of %3f with %d features added and C=%f'%(besta.max(),(bestoveralla+1),bestcforbesta))

bestfeats = [f527col,f528col]+sortedfeatinds[:(bestoveralla+1)].tolist()

trfinal=trdata[:,tuple(bestfeats)]
lr.C=bestcforbesta
lr.fit(trfinal[tr],labels2[tr])
a=auc_score(labels2[te],lr.predict_proba(trfinal[te])[:,1])
print('Checking AUC: %3f'%a)

cPickle.dump(tuple(bestfeats),open('bestfeatures.cp','wb'))
cPickle.dump(lr,open('classmodel.cp','wb'))

thresh=np.linspace(0,1,100)
f1=np.zeros_like(thresh)
for i in thresh:
    f1[count]=f1_score(labels2[te],lr.predict_proba(trfinal[te])[:,1]>i)
    count+=1
bestthresh=thresh[f1.argmax()]
cPickle.dump(bestthresh,open('bestf1.cp','wb'))