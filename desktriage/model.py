import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,average_precision_score,confusion_matrix,f1_score,
                             precision_score,recall_score,roc_auc_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

def fit_models(texts,y,seed=42):
    baseline=make_pipeline(CountVectorizer(ngram_range=(1,1),min_df=1),MultinomialNB(alpha=.5))
    stronger=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True),
                           LogisticRegression(max_iter=1000,solver="liblinear",class_weight="balanced",random_state=seed))
    baseline.fit(texts,y); stronger.fit(texts,y)
    return {"baseline_count_nb":baseline,"stronger_tfidf_logistic":stronger}

def evaluate(model,texts,y):
    y=np.asarray(y); pred=model.predict(texts); labels=np.unique(y)
    out={"n":int(len(y)),"accuracy":round(float(accuracy_score(y,pred)),4),
         "macro_f1":round(float(f1_score(y,pred,average="macro",zero_division=0)),4),
         "confusion_matrix":confusion_matrix(y,pred,labels=labels).tolist(),
         "labels":[str(v) for v in labels],"error_count":int(np.sum(pred!=y))}
    if len(labels)==2:
        p=model.predict_proba(texts)[:,1]
        out.update({"roc_auc":round(float(roc_auc_score(y,p)),4),"average_precision":round(float(average_precision_score(y,p)),4),
                    "precision":round(float(precision_score(y,pred,zero_division=0)),4),"recall":round(float(recall_score(y,pred,zero_division=0)),4),"f1_at_default_threshold":round(float(f1_score(y,pred,zero_division=0)),4)})
    return out

def subgroup_metrics(model,texts,y,groups):
    y=np.asarray(y);pred=model.predict(texts);groups=np.asarray(groups);out={};rates=[];recalls=[]
    for g in np.unique(groups):
        m=groups==g
        rate=float(np.mean(pred[m]==1));rec=float(recall_score(y[m],pred[m],zero_division=0))
        out[str(g)]={"n":int(m.sum()),"label_rate":round(float(np.mean(y[m]==1)),4),"prediction_rate":round(rate,4),"recall":round(rec,4)}
        rates.append(rate);recalls.append(rec)
    return {"groups":out,"prediction_rate_gap":round(max(rates)-min(rates),4),"recall_gap":round(max(recalls)-min(recalls),4)}
