"""Bounded temporal research splits and interval leakage checks; no fitting/I/O.

Indices identify an already chronologically ordered dataset, not exchange days.
Callers must choose session/bar granularity and retain calendar/data provenance.
Purged CV can include later training intervals; it is not a live causal replay.
Use walk-forward folds plus label-availability cutoffs for sequential research.
"""
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime,timedelta
import re

from .data import utc_time


class ValidationError(ValueError):pass


def _time(value):
    try:return utc_time(value)
    except Exception:raise ValidationError('aware_timestamp_required') from None


@dataclass(frozen=True)
class LabelInterval:
    identity:str
    starts_at:datetime
    ends_at:datetime
    available_at:datetime

    def __post_init__(self):
        for name in ('starts_at','ends_at','available_at'):object.__setattr__(self,name,_time(getattr(self,name)))
        if (not isinstance(self.identity,str) or not re.fullmatch('[0-9a-f]{64}',self.identity)
                or not self.starts_at<=self.ends_at<=self.available_at):
            raise ValidationError('invalid_label_interval')


def _indices(values,length):
    if (type(values) not in (list,tuple) or len(values)>100000
            or any(type(v) is not int or not 0<=v<length for v in values) or len(set(values))!=len(values)):
        raise ValidationError('invalid_or_duplicate_indices')
    return tuple(values)


def purged_training(labels,candidate_indices,test_indices,*,cutoff,embargo):
    at=_time(cutoff)
    if (type(labels) not in (tuple,list) or not 1<=len(labels)<=100000
            or any(type(v) is not LabelInterval for v in labels)
            or len({v.identity for v in labels})!=len(labels)
            or any(a.starts_at>b.starts_at for a,b in zip(labels,labels[1:]))
            or type(embargo) is not timedelta or not timedelta(0)<=embargo<=timedelta(days=30)):
        raise ValidationError('invalid_temporal_dataset_or_embargo')
    candidates=_indices(candidate_indices,len(labels));test=_indices(test_indices,len(labels))
    if not test or set(candidates)&set(test):raise ValidationError('disjoint_test_indices_required')
    try:windows=sorted((labels[i].starts_at,labels[i].ends_at+embargo) for i in test)
    except OverflowError:raise ValidationError('embargo_timestamp_overflow') from None
    merged=[]
    for start,end in windows:
        if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    starts=[w[0] for w in merged];result=[]
    for i in candidates:
        label=labels[i]
        if label.available_at>at:continue
        j=bisect_right(starts,label.ends_at)-1
        if j>=0 and merged[j][1]>=label.starts_at:continue
        result.append(i)
    return tuple(result)


@dataclass(frozen=True)
class Fold:
    train:tuple
    validation:tuple
    test:tuple
    excluded:tuple


def walk_forward(length,*,train,validation,test,gap,step,expanding=False):
    if (any(type(v) is not int for v in (length,train,validation,test,gap,step))
            or not 1<=length<=100000 or train<2 or validation<1 or test<1 or gap<0
            or step<test or type(expanding) is not bool):
        raise ValidationError('invalid_walk_forward_dimensions')
    width=train+validation+test+2*gap
    if width>length:raise ValidationError('insufficient_complete_fold_data')
    count=(length-width)//step+1
    # Bound both fold count and expanded member allocation before materializing.
    members=count*width+(step*count*(count-1)//2 if expanding else 0)
    if count>1000 or members>1000000:raise ValidationError('walk_forward_compute_budget')
    folds=[]
    for start in range(0,length-width+1,step):
        end_train=start+train;start_validation=end_train+gap
        end_validation=start_validation+validation;start_test=end_validation+gap
        folds.append(Fold(tuple(range(0 if expanding else start,end_train)),
            tuple(range(start_validation,end_validation)),tuple(range(start_test,start_test+test)),
            tuple(range(end_train,start_validation))+tuple(range(end_validation,start_test))))
    return tuple(folds)
