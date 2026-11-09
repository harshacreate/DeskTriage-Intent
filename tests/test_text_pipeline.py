import json,os,subprocess,sys
import numpy as np
from desktriage.data import GROUP_SPLIT,make_dataset
from desktriage.split import split_indices
from desktriage.train import run
def test_text_data_is_deterministic_and_valid():
    x,y,groups,diag=make_dataset(300,12);x2,y2,_,_=make_dataset(300,12)
    assert len(x)==len(y)==300 and all(isinstance(v,str) and v for v in x)
    assert np.array_equal(x,x2) and np.array_equal(y,y2)
def test_split_strategy():
    x,y,groups,diag=make_dataset(600,8);tr,te=split_indices(y,3,groups if GROUP_SPLIT else None)
    assert not set(tr).intersection(te)
    if GROUP_SPLIT: assert not set(groups[tr]).intersection(groups[te])
def test_models_metrics_and_cli():
    report=run(9,480);assert report["n_test"]>0
    assert 0<=report["stronger_model"]["macro_f1"]<=1
    env=dict(os.environ);env["PYTHONPATH"]=os.pathsep.join([os.getcwd(),env.get("PYTHONPATH","")])
    r=subprocess.run([sys.executable,"-m","desktriage.train","--samples","320","--json"],check=True,capture_output=True,text=True,env=env)
    assert json.loads(r.stdout)["project"]=='DeskTriage-Intent — Support Ticket Intent Classifier'
