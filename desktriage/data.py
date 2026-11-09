"""Synthetic text dataset for DeskTriage-Intent — Support Ticket Intent Classifier; no real applicant or customer text is used."""
from __future__ import annotations
import numpy as np

GROUP_SPLIT = False
GROUP_NAME = None

def make_dataset(n=1600, seed=42):
    if n < 120: raise ValueError("n must be at least 120")
    rng=np.random.default_rng(seed)
    intents={"billing":["invoice","charge","payment","duplicate","receipt","subscription"],"account_access":["password","login","locked","verify","account","reset"],"refund":["refund","return","reimbursement","money back","cancelled"],"order_status":["order","tracking","status","delivered","package","shipment"],"technical_issue":["error","crash","bug","screen","update","connection"],"shipping":["shipping","delivery","address","courier","late","parcel"],"cancellation":["cancel","stop","renewal","close","unsubscribe","plan"]}
    labels=np.array(list(intents),dtype=object); y=rng.choice(labels,size=n)
    openers=["please help", "I need assistance", "could you check", "I am contacting support", "can someone explain"]
    closers=["thanks", "this is urgent", "please advise", "I tried twice", "what should I do"]
    noise=["today", "again", "online", "mobile", "yesterday", "customer", "service", "app"]
    texts=[]
    for label in y:
        words=list(intents[label]);rng.shuffle(words);chosen=words[:int(rng.integers(2,min(5,len(words))+1))]
        texts.append(" ".join([rng.choice(openers),*chosen,*rng.choice(noise,size=2,replace=False),rng.choice(closers)]))
    split_groups=None;diagnostic_groups=None
    if len(texts)!=n or np.asarray(y).shape!=(n,): raise ValueError("invalid generated text dataset")
    if any(not isinstance(x,str) or not x.strip() for x in texts): raise ValueError("text rows must be non-empty strings")
    return np.asarray(texts,dtype=object), np.asarray(y), None if split_groups is None else np.asarray(split_groups), None if diagnostic_groups is None else np.asarray(diagnostic_groups)
