import argparse,json
from .data import GROUP_NAME,GROUP_SPLIT,make_dataset
from .split import split_indices
from .model import fit_models,evaluate,subgroup_metrics
PROJECT='DeskTriage-Intent — Support Ticket Intent Classifier'
def run(seed=42,samples=1600):
    texts,y,split_groups,diagnostic_groups=make_dataset(samples,seed)
    tr,te=split_indices(y,seed,split_groups if GROUP_SPLIT else None)
    models=fit_models(texts[tr],y[tr],seed)
    report={"project":PROJECT,"data":"synthetic demonstration text","seed":int(seed),"split":"candidate-group holdout" if GROUP_SPLIT else "stratified random holdout",
            "n_train":int(len(tr)),"n_test":int(len(te)),"baseline":evaluate(models["baseline_count_nb"],texts[te],y[te]),
            "stronger_model":evaluate(models["stronger_tfidf_logistic"],texts[te],y[te])}
    if diagnostic_groups is not None:
        report["subgroup_diagnostic"]={"attribute":GROUP_NAME,**subgroup_metrics(models["stronger_tfidf_logistic"],texts[te],y[te],diagnostic_groups[te])}
    return report
def main(argv=None):
    p=argparse.ArgumentParser(description="Train and evaluate a synthetic text prototype.");p.add_argument("--seed",type=int,default=42);p.add_argument("--samples",type=int,default=1600);p.add_argument("--json",action="store_true");a=p.parse_args(argv);print(json.dumps(run(a.seed,a.samples),indent=2))
if __name__=="__main__":main()
