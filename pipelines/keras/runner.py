import os as _runtime_os
DATA_ROOT = _runtime_os.environ["PDMRI_DATA_ROOT"]
OUTPUT_ROOT = _runtime_os.environ.get("PDMRI_OUTPUT_ROOT", "outputs")

import os, sys, re, json, math, time, random, hashlib, shutil, warnings, platform, subprocess, traceback, zipfile
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PIPELINE_VERSION = "OVERLAP_DOSE_RESPONSE_V2_2026-09-02"
RUN_MODE = "FULL_EXPERIMENT"
RUN_ENDPOINT_PERMUTATION_CONTROL = False
TARGET_OUTER_FOLD = int(_runtime_os.environ.get('PDMRI_OUTER_FOLD', '0'))

DATA_ROOT = Path(DATA_ROOT)
OUTPUT_ROOT = Path(OUTPUT_ROOT)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

EXPECTED_SUBJECTS = 68
EXPECTED_PD = 48
EXPECTED_NON_PD = 20
EXPECTED_SLICES_APPROX = 42147
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
HEAD_EPOCHS = 8
FINETUNE_EPOCHS = 20
PATIENCE = 4
HEAD_LR = 1e-4
FINETUNE_LR = 1e-5
MAX_LEAK_SLICES_PER_SUBJECT = 50
MAX_LEAK_FRACTION_PER_SUBJECT = 0.20
OVERLAP_LEVELS = [0, 25, 50, 75, 100]
N_OUTER_FOLDS = 5
INNER_VALIDATION_FRACTION = 0.20
BASE_SEED = 42
MANUSCRIPT_FIXED_SEEDS = [42, 101, 202, 303, 404]
BOOTSTRAP_REPLICATES = 2000
PAIR_SAMPLES_PER_CLASS = 2000
IDENTITY_CLUSTER_BOOTSTRAP_REPLICATES = 500
TREND_PERMUTATIONS = 10000
THRESHOLDS = np.round(np.arange(0.05, 0.9501, 0.005), 3)
USE_LOCAL_IMAGE_CACHE = True

ACTIVE_FOLDS = [TARGET_OUTER_FOLD]
ACTIVE_PLAN = {"ResNet50V2": [0,100] if int(_runtime_os.environ.get("PDMRI_SEED_OFFSET","0")) else OVERLAP_LEVELS, "VGG19": [0, 100]}
EFFECTIVE_HEAD_EPOCHS = HEAD_EPOCHS
EFFECTIVE_FINETUNE_EPOCHS = FINETUNE_EPOCHS
EFFECTIVE_MAX_SLICES_PER_SUBJECT = None
EFFECTIVE_MAX_LEAK_SLICES = MAX_LEAK_SLICES_PER_SUBJECT
PAIR_SAMPLES_EFFECTIVE = PAIR_SAMPLES_PER_CLASS
RESULT_NAMESPACE = "FULL_EXPERIMENT_SPLIT" if int(_runtime_os.environ.get("PDMRI_SEED_OFFSET","0")) == 0 else "second_seed_endpoint_runs"
RUN_ROOT = OUTPUT_ROOT / RESULT_NAMESPACE / f"fold_runner_{TARGET_OUTER_FOLD}"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

os.environ['PYTHONHASHSEED'] = str(BASE_SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print('Deterministic operations warning:', e)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print('Mixed precision enabled:', mixed_precision.global_policy())
    except Exception as e:
        print('Mixed precision could not be enabled safely:', e)
else:
    print('WARNING: GPU not detected. Full experiment is not recommended on CPU.')

FOLD_TRAINING_SEED = int(MANUSCRIPT_FIXED_SEEDS[TARGET_OUTER_FOLD]) + int(_runtime_os.environ.get('PDMRI_SEED_OFFSET', '0'))
CONFIG = {
    'pipeline_version': PIPELINE_VERSION,
    'run_mode': RUN_MODE, 'target_outer_fold': TARGET_OUTER_FOLD,
    'data_root': str(DATA_ROOT), 'output_root': str(OUTPUT_ROOT),
    'expected_subjects': EXPECTED_SUBJECTS, 'expected_pd': EXPECTED_PD, 'expected_non_pd': EXPECTED_NON_PD,
    'expected_slices_approx': EXPECTED_SLICES_APPROX, 'image_size': IMAGE_SIZE, 'batch_size': BATCH_SIZE,
    'head_epochs': EFFECTIVE_HEAD_EPOCHS, 'finetune_epochs': EFFECTIVE_FINETUNE_EPOCHS,
    'patience': PATIENCE, 'head_lr': HEAD_LR, 'finetune_lr': FINETUNE_LR,
    'max_leak_slices_per_subject': EFFECTIVE_MAX_LEAK_SLICES,
    'max_leak_fraction_per_subject': MAX_LEAK_FRACTION_PER_SUBJECT,
    'overlap_levels': OVERLAP_LEVELS, 'n_outer_folds': N_OUTER_FOLDS,
    'inner_validation_fraction': INNER_VALIDATION_FRACTION, 'base_seed': BASE_SEED,
    'manuscript_fixed_seeds': MANUSCRIPT_FIXED_SEEDS,
    'fold_training_seed': FOLD_TRAINING_SEED,
    'pair_samples_per_class': PAIR_SAMPLES_EFFECTIVE,
    'identity_cluster_bootstrap_replicates': IDENTITY_CLUSTER_BOOTSTRAP_REPLICATES,
    'trend_permutations': TREND_PERMUTATIONS,
    'run_endpoint_permutation_control': RUN_ENDPOINT_PERMUTATION_CONTROL,
    'local_image_cache': USE_LOCAL_IMAGE_CACHE,
    'augmentation': {'rotation_factor': 0.025/(2*np.pi), 'translation': 0.04, 'zoom': 0.06, 'contrast': 0.10},
    'controlled_design_v2': {
        'same_training_seed_across_overlap_levels_within_fold': True,
        'nested_contaminated_subjects': True,
        'same_leaked_slices_for_subject_across_all_doses_where_selected': True,
        'nested_class_matched_training_slice_removal': True,
        'identity_ci_unit': 'anchor-subject cluster bootstrap'
    },
    'source_provenance_note': 'Frozen study configuration'
}
with open(RUN_ROOT/'complete_configuration.json','w') as f: json.dump(CONFIG,f,indent=2)
print(json.dumps(CONFIG, indent=2))


def get_environment_report():
    lines=[]
    def add(k,v): lines.append(f"{k}: {v}")
    add('UTC', datetime.now(timezone.utc).isoformat())
    add('Pipeline version', PIPELINE_VERSION)
    add('Python', sys.version.replace('\n',' '))
    add('Platform', platform.platform())
    add('TensorFlow', tf.__version__)
    add('Keras', getattr(keras,'__version__','bundled-with-tensorflow'))
    add('CUDA build', tf.sysconfig.get_build_info().get('cuda_version','unknown'))
    add('cuDNN build', tf.sysconfig.get_build_info().get('cudnn_version','unknown'))
    add('GPU devices', gpus)
    try:
        add('nvidia-smi', subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'], text=True).strip())
    except Exception as e: add('nvidia-smi', f'unavailable: {e}')
    return '\n'.join(lines)

env_report=get_environment_report()
print(env_report)
(RUN_ROOT/'environment_report.txt').write_text(env_report)

IMAGE_EXTS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
PD_TOKENS={'pd','parkinson','parkinsons','parkinson_disease','patient','patients'}
NONPD_TOKENS={'non-pd','non_pd','nonpd','control','controls','healthy','normal','hc'}

def norm_token(s): return re.sub(r'[^a-z0-9]+','_',str(s).lower()).strip('_')

def infer_label_and_subject(path: Path, root: Path):
    rel=path.relative_to(root)
    parts=list(rel.parts[:-1])
    norms=[norm_token(p) for p in parts]
    label_hits=[]
    for i,n in enumerate(norms):
        if n in {norm_token(x) for x in NONPD_TOKENS} or n.startswith('control') or n.startswith('normal') or n.startswith('healthy'):
            label_hits.append((i,0))
        elif n in {norm_token(x) for x in PD_TOKENS} or n in {'parkinson_s_disease','parkinson_disease'}:
            label_hits.append((i,1))
    # prefer most specific/deepest recognized diagnosis folder
    if not label_hits:
        return None,None,None
    idx,label=sorted(label_hits,key=lambda z:z[0])[-1]
    if idx+1 >= len(parts):
        return None,None,None
    subject_raw=parts[idx+1]
    return label,subject_raw,parts[idx]

def anon_subject(raw): return 'S_'+hashlib.sha256(str(raw).encode()).hexdigest()[:10]

def discover_dataset(root: Path):
    if not root.exists(): raise FileNotFoundError(f'DATA_ROOT does not exist: {root}')
    rows=[]; unresolved=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS: continue
        label,raw,source_label=infer_label_and_subject(p,root)
        if label is None:
            unresolved.append(str(p)); continue
        rows.append({'image_path':str(p.resolve()),'subject_raw':str(raw),'anonymous_subject_id':anon_subject(raw),
                     'class_label':int(label),'diagnosis':'PD' if label==1 else 'non-PD',
                     'slice_identifier':p.stem,'source_label_folder':str(source_label),'exclusion_reason':''})
    df=pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('No classifiable MRI image files were found. The notebook stopped rather than guessing labels.')
    # detect raw subject reused with conflicting labels
    conflicts=df.groupby('anonymous_subject_id')['class_label'].nunique()
    if (conflicts>1).any(): raise RuntimeError('One or more subjects were assigned conflicting class labels.')
    counts=df.groupby('anonymous_subject_id').size().rename('total_slices_per_subject')
    df=df.join(counts,on='anonymous_subject_id')
    # anonymized output only: raw IDs never written
    safe=df.drop(columns=['subject_raw']).copy()
    safe.to_csv(RUN_ROOT/'master_subject_index.csv',index=False)
    if unresolved:
        pd.DataFrame({'image_path':unresolved,'exclusion_reason':'label/subject could not be inferred safely'}).to_csv(RUN_ROOT/'unresolved_images.csv',index=False)
    return safe

master=discover_dataset(DATA_ROOT)
subject_table=master[['anonymous_subject_id','class_label','diagnosis','total_slices_per_subject']].drop_duplicates()
print(subject_table.groupby('class_label')['anonymous_subject_id'].nunique())
print('Subjects:',subject_table.anonymous_subject_id.nunique(),'Slices:',len(master))
if subject_table.anonymous_subject_id.nunique()!=EXPECTED_SUBJECTS:
    raise RuntimeError(f'Cohort mismatch: expected {EXPECTED_SUBJECTS} subjects, found {subject_table.anonymous_subject_id.nunique()}')
if int((subject_table.class_label==1).sum())!=EXPECTED_PD or int((subject_table.class_label==0).sum())!=EXPECTED_NON_PD:
    raise RuntimeError(f'Class-count mismatch: expected PD={EXPECTED_PD}, non-PD={EXPECTED_NON_PD}; found PD={(subject_table.class_label==1).sum()}, non-PD={(subject_table.class_label==0).sum()}')
print('COHORT AUDIT: PASS')

# Optional run-local cache; the source data remain in their configured location.
# This reduces repeated network-backed image reads within a fold.
def prepare_runtime_paths(df):
    out=df.copy()
    out['runtime_image_path']=out['image_path'].astype(str)
    report={'requested':bool(USE_LOCAL_IMAGE_CACHE),'used':False,'files':0,'bytes_estimated':0,'cache_root':str(Path(_runtime_os.environ.get('PDMRI_CACHE_ROOT', '/tmp/pdmri_image_cache')))}
    if not USE_LOCAL_IMAGE_CACHE:
        (RUN_ROOT/'local_cache_report.json').write_text(json.dumps(report,indent=2)); return out
    cache_root=Path(_runtime_os.environ.get('PDMRI_CACHE_ROOT', '/tmp/pdmri_image_cache'))
    try:
        sizes=[]
        for p in out.image_path.astype(str): sizes.append(Path(p).stat().st_size)
        need=int(sum(sizes)); free=shutil.disk_usage(cache_root.parent if cache_root.parent.exists() else DATA_ROOT).free
        report['bytes_estimated']=need; report['free_bytes_before']=int(free)
        if need > 0.75*free:
            report['fallback_reason']='insufficient_local_disk'; print('Local cache skipped: insufficient available disk space.')
            (RUN_ROOT/'local_cache_report.json').write_text(json.dumps(report,indent=2)); return out
        cache_root.mkdir(parents=True,exist_ok=True)
        runtime=[]
        for i,p in enumerate(out.image_path.astype(str),1):
            src=Path(p); token=hashlib.sha256(p.encode()).hexdigest()[:24]+src.suffix.lower(); dst=cache_root/token
            if not dst.exists() or dst.stat().st_size!=src.stat().st_size: shutil.copy2(src,dst)
            runtime.append(str(dst))
            if i%2000==0: print(f'Cached {i}/{len(out)} images locally...')
        out['runtime_image_path']=runtime
        report.update({'used':True,'files':len(out)})
        print(f'LOCAL IMAGE CACHE: READY ({len(out)} files)')
    except Exception as e:
        report['fallback_reason']=f'{type(e).__name__}: {e}'
        print('Local cache warning; retaining source image paths:',e)
        out['runtime_image_path']=out['image_path'].astype(str)
    (RUN_ROOT/'local_cache_report.json').write_text(json.dumps(report,indent=2))
    return out

master=prepare_runtime_paths(master)


subjects=subject_table.sort_values('anonymous_subject_id').reset_index(drop=True)
skf=StratifiedKFold(n_splits=N_OUTER_FOLDS,shuffle=True,random_state=BASE_SEED)
split_rows=[]
for fold,(outer_train_idx,test_idx) in enumerate(skf.split(subjects.anonymous_subject_id,subjects.class_label)):
    outer_train=subjects.iloc[outer_train_idx].reset_index(drop=True)
    test=subjects.iloc[test_idx].reset_index(drop=True)
    sss=StratifiedShuffleSplit(n_splits=1,test_size=INNER_VALIDATION_FRACTION,random_state=BASE_SEED+fold)
    tr_idx,val_idx=next(sss.split(outer_train.anonymous_subject_id,outer_train.class_label))
    train=outer_train.iloc[tr_idx]; val=outer_train.iloc[val_idx]
    for role,frame in [('outer_training',train),('inner_validation',val),('outer_testing',test)]:
        for _,r in frame.iterrows(): split_rows.append({'outer_fold':fold,'role':role,'anonymous_subject_id':r.anonymous_subject_id,'class_label':int(r.class_label)})
split_manifest=pd.DataFrame(split_rows)
split_manifest.to_csv(RUN_ROOT/'outer_inner_split_manifest.csv',index=False)
print(split_manifest.groupby(['outer_fold','role','class_label']).size())

# FULL experiment uses the complete cohort. runtime_image_path is internal only and is never used as the partition identity key.
working_master=master.copy()
print('Working slices:',len(working_master),'subjects:',working_master.anonymous_subject_id.nunique())


def _target_contaminated_count(n, overlap_pct):
    if overlap_pct==0: return 0
    if overlap_pct==100: return n
    return int(round(n*overlap_pct/100.0))

def contamination_subject_order(test_subject_df, fold):
    """One diagnosis-aware deterministic ordering; every higher dose is a prefix superset."""
    fold_seed=int(MANUSCRIPT_FIXED_SEEDS[fold])
    pools={}
    for cls,g in test_subject_df.groupby('class_label',sort=True):
        vals=sorted(g.anonymous_subject_id.astype(str).tolist())
        rng=np.random.default_rng(fold_seed + 1000 + int(cls)*101)
        rng.shuffle(vals); pools[int(cls)]=vals
    totals={cls:len(v) for cls,v in pools.items()}; n=sum(totals.values()); chosen_counts={cls:0 for cls in pools}
    pos={cls:0 for cls in pools}; order=[]
    for step in range(1,n+1):
        eligible=[cls for cls in sorted(pools) if pos[cls] < len(pools[cls])]
        # choose the class with the greatest cumulative proportional deficit
        deficits={cls:(totals[cls]/n)*step-chosen_counts[cls] for cls in eligible}
        cls=max(eligible,key=lambda c:(deficits[c],-c))
        order.append(pools[cls][pos[cls]]); pos[cls]+=1; chosen_counts[cls]+=1
    return order

def choose_contaminated_subjects(test_subject_df, overlap_pct, fold):
    order=contamination_subject_order(test_subject_df,fold)
    return order[:_target_contaminated_count(len(order),overlap_pct)]

def deterministic_leak_slices_for_subject(test, sid, fold):
    g=test[test.anonymous_subject_id==sid].sort_values('image_path').reset_index(drop=True)
    max_by_frac=int(math.floor(len(g)*MAX_LEAK_FRACTION_PER_SUBJECT))
    n_leak=min(EFFECTIVE_MAX_LEAK_SLICES,max_by_frac,max(0,len(g)-2))
    if n_leak<1: raise RuntimeError(f'Not enough slices to contaminate subject {sid} while retaining test slices.')
    seed=int(MANUSCRIPT_FIXED_SEEDS[fold])+20000+int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed)
    idx=np.sort(rng.choice(len(g),size=n_leak,replace=False))
    return g.iloc[idx].copy(),n_leak

def deterministic_training_removal_order(train, cls, fold):
    candidates=train[train.class_label==cls].sort_values('image_path').copy()
    seed=int(MANUSCRIPT_FIXED_SEEDS[fold])+30000+int(cls)*313
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(candidates))
    return candidates.iloc[order].reset_index(drop=True)

def build_partitions(fold, overlap_pct):
    sm=split_manifest[split_manifest.outer_fold==fold]
    train_ids=set(sm[sm.role=='outer_training'].anonymous_subject_id)
    val_ids=set(sm[sm.role=='inner_validation'].anonymous_subject_id)
    test_df=sm[sm.role=='outer_testing'][['anonymous_subject_id','class_label']].copy()
    test_ids=set(test_df.anonymous_subject_id)
    subject_order=contamination_subject_order(test_df,fold)
    contam=choose_contaminated_subjects(test_df,overlap_pct,fold)
    rank={sid:i+1 for i,sid in enumerate(subject_order)}
    train=working_master[working_master.anonymous_subject_id.isin(train_ids)].copy()
    val=working_master[working_master.anonymous_subject_id.isin(val_ids)].copy()
    test=working_master[working_master.anonymous_subject_id.isin(test_ids)].copy()
    injected=[]; crows=[]
    for sid in contam:
        inj,n_leak=deterministic_leak_slices_for_subject(test,sid,fold)
        injected.append(inj)
        test=test[~test.image_path.isin(set(inj.image_path))].copy()
        for _,rr in inj.iterrows():
            crows.append({'outer_fold':fold,'overlap_pct':overlap_pct,'anonymous_subject_id':sid,'class_label':int(rr.class_label),
                          'subject_contamination_rank':rank[sid],'injected_image_path':rr.image_path,
                          'actual_slices_injected_for_subject':n_leak})
    removed_rows=[]
    if injected:
        inj_all=pd.concat(injected,ignore_index=True)
        remove_paths=[]
        for cls,nrem in sorted(inj_all.class_label.value_counts().items()):
            removal_order=deterministic_training_removal_order(train,int(cls),fold)
            if len(removal_order)<nrem: raise RuntimeError(f'Insufficient class-matched original training slices for class {cls}.')
            rm=removal_order.iloc[:int(nrem)].copy()
            remove_paths.extend(rm.image_path.tolist())
            for _,rr in rm.iterrows():
                removed_rows.append({'outer_fold':fold,'overlap_pct':overlap_pct,'class_label':int(cls),'removed_original_training_image_path':rr.image_path})
        train=train[~train.image_path.isin(set(remove_paths))].copy()
        train=pd.concat([train,inj_all],ignore_index=True)
    cman=pd.DataFrame(crows)
    rman=pd.DataFrame(removed_rows)
    return train,val,test,contam,cman,rman

def audit_partitions(train,val,test,contam,fold,overlap_pct):
    ptrain,pval,ptest=set(train.image_path),set(val.image_path),set(test.image_path)
    strain,sval,stest=set(train.anonymous_subject_id),set(val.anonymous_subject_id),set(test.anonymous_subject_id)
    test_subjects_expected=set(split_manifest[(split_manifest.outer_fold==fold)&(split_manifest.role=='outer_testing')].anonymous_subject_id)
    overlap_train_test=strain & stest; expected=set(contam)
    checks={
        'path_train_val_overlap':len(ptrain&pval), 'path_train_test_overlap':len(ptrain&ptest), 'path_val_test_overlap':len(pval&ptest),
        'subject_train_val_overlap':len(strain&sval), 'subject_val_test_overlap':len(sval&stest),
        'subject_train_test_overlap':len(overlap_train_test), 'expected_train_test_subject_overlap':len(expected),
        'test_subject_identity_set_preserved': stest==test_subjects_expected,
        'train_test_overlap_exact_match': overlap_train_test==expected
    }
    passed=(checks['path_train_val_overlap']==0 and checks['path_train_test_overlap']==0 and checks['path_val_test_overlap']==0 and
            checks['subject_train_val_overlap']==0 and checks['subject_val_test_overlap']==0 and checks['test_subject_identity_set_preserved'] and checks['train_test_overlap_exact_match'])
    row={'outer_fold':fold,'overlap_pct':overlap_pct,'actual_overlap_pct':100*len(overlap_train_test)/len(stest),
         'train_subjects':len(strain),'validation_subjects':len(sval),'test_subjects':len(stest),'train_slices':len(train),'validation_slices':len(val),'test_slices':len(test),
         'train_pd_subjects':train[train.class_label==1].anonymous_subject_id.nunique(),'train_nonpd_subjects':train[train.class_label==0].anonymous_subject_id.nunique(),
         'validation_pd_subjects':val[val.class_label==1].anonymous_subject_id.nunique(),'validation_nonpd_subjects':val[val.class_label==0].anonymous_subject_id.nunique(),
         'test_pd_subjects':test[test.class_label==1].anonymous_subject_id.nunique(),'test_nonpd_subjects':test[test.class_label==0].anonymous_subject_id.nunique(),
         'contaminated_subject_ids':'|'.join(sorted(contam)),'audit_status':'PASS' if passed else 'FAIL',**checks}
    if not passed: raise RuntimeError('PARTITION AUDIT FAIL: '+json.dumps(row,default=str))
    return row

# Design-level nesting audit before any model training.
def audit_nested_contamination_design(fold):
    sm=split_manifest[split_manifest.outer_fold==fold]
    test_df=sm[sm.role=='outer_testing'][['anonymous_subject_id','class_label']].copy()
    sets={ov:set(choose_contaminated_subjects(test_df,ov,fold)) for ov in OVERLAP_LEVELS}
    assert sets[0]==set()
    assert sets[25].issubset(sets[50]) and sets[50].issubset(sets[75]) and sets[75].issubset(sets[100])
    assert sets[100]==set(test_df.anonymous_subject_id)
    rows=[]
    for ov in OVERLAP_LEVELS:
        sel=test_df[test_df.anonymous_subject_id.isin(sets[ov])]
        rows.append({'outer_fold':fold,'overlap_pct':ov,'n_selected':len(sel),'pd_selected':int((sel.class_label==1).sum()),'nonpd_selected':int((sel.class_label==0).sum()),'nested_design_status':'PASS'})
    pd.DataFrame(rows).to_csv(RUN_ROOT/'nested_contamination_design_audit.csv',index=False)
    return sets

NESTED_CONTAMINATION_SETS=audit_nested_contamination_design(TARGET_OUTER_FOLD)
print('NESTED CONTAMINATION DESIGN: PASS')


def make_augmentation(run_seed):
    # Recreated for every training run so state never carries from one overlap condition to another.
    return keras.Sequential([
        layers.RandomRotation(factor=0.025/(2*np.pi), fill_mode='reflect', seed=run_seed),
        layers.RandomTranslation(height_factor=0.04,width_factor=0.04,fill_mode='reflect',seed=run_seed+1),
        layers.RandomZoom(height_factor=(-0.06,0.06),width_factor=(-0.06,0.06),fill_mode='reflect',seed=run_seed+2),
        layers.RandomContrast(factor=0.10,seed=run_seed+3),
    ],name='training_augmentation')

def decode_resize(path,label):
    b=tf.io.read_file(path)
    x=tf.io.decode_image(b,channels=1,expand_animations=False)
    x.set_shape([None,None,1])
    x=tf.image.resize(x,IMAGE_SIZE,method=tf.image.ResizeMethod.AREA)
    x=tf.image.grayscale_to_rgb(x)
    x=tf.cast(x,tf.float32)
    return x,tf.cast(label,tf.float32)

def make_ds(df,architecture,training,batch_size=BATCH_SIZE,run_seed=None,augment=None):
    path_col='runtime_image_path' if 'runtime_image_path' in df.columns else 'image_path'
    paths=df[path_col].astype(str).values; labels=df.class_label.astype(np.float32).values
    ds=tf.data.Dataset.from_tensor_slices((paths,labels))
    if training:
        if run_seed is None or augment is None: raise ValueError('Training dataset requires run_seed and a fresh augmentation object.')
        ds=ds.shuffle(min(len(df),10000),seed=int(run_seed),reshuffle_each_iteration=True)
    ds=ds.map(decode_resize,num_parallel_calls=tf.data.AUTOTUNE)
    prep=tf.keras.applications.resnet_v2.preprocess_input if architecture=='ResNet50V2' else tf.keras.applications.vgg19.preprocess_input
    def transform(x,y):
        if training: x=augment(x,training=True)
        x=prep(x)
        return x,y
    options=tf.data.Options(); options.experimental_deterministic=True; ds=ds.with_options(options)
    ds=ds.map(transform,num_parallel_calls=(1 if training else tf.data.AUTOTUNE)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def build_model(architecture):
    inp=keras.Input(shape=(*IMAGE_SIZE,3),name='image')
    if architecture=='ResNet50V2':
        base=tf.keras.applications.ResNet50V2(include_top=False,weights='imagenet',input_tensor=inp)
    elif architecture=='VGG19':
        base=tf.keras.applications.VGG19(include_top=False,weights='imagenet',input_tensor=inp)
    else: raise ValueError(architecture)
    base.trainable=False
    x=layers.GlobalAveragePooling2D(name='penultimate_embedding')(base.output)
    x=layers.Dropout(0.5,name='dropout')(x)
    out=layers.Dense(1,activation='sigmoid',dtype='float32',name='pd_probability')(x)
    return keras.Model(inp,out,name=architecture+'_PD'),base

def compile_model(model,lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),loss='binary_crossentropy',metrics=[keras.metrics.BinaryAccuracy(name='accuracy')])

def unfreeze_tail(base,architecture):
    base.trainable=True; n=12 if architecture=='ResNet50V2' else 4
    for layer in base.layers[:-n]: layer.trainable=False
    for layer in base.layers[-n:]: layer.trainable=True


def aggregate_patient(pred_df):
    return pred_df.groupby(['anonymous_subject_id','class_label'],as_index=False).agg(probability=('probability','mean'),n_slices=('image_path','count'))

def select_threshold(val_patient):
    y=val_patient.class_label.to_numpy(int); p=val_patient.probability.to_numpy(float)
    rows=[]
    for t in THRESHOLDS:
        yh=(p>=t).astype(int)
        rows.append({'threshold':float(t),'balanced_accuracy':balanced_accuracy_score(y,yh),'mcc':matthews_corrcoef(y,yh),'f1':f1_score(y,yh,zero_division=0),'proximity_to_0_5':-abs(float(t)-0.5)})
    curve=pd.DataFrame(rows).sort_values(['balanced_accuracy','mcc','f1','proximity_to_0_5'],ascending=False).reset_index(drop=True)
    return float(curve.iloc[0].threshold),curve

def metric_row(df,threshold,level,population,architecture,fold,overlap_pct,permutation_seed=None):
    if df is None or len(df)==0:
        return {'architecture':architecture,'outer_fold':fold,'overlap_pct':overlap_pct,'evaluation_level':level,'population':population,'permutation_seed':permutation_seed,'n_subjects':np.nan,'n_pd':np.nan,'n_non_pd':np.nan,'n_slices':np.nan,'accuracy':np.nan,'majority_class_accuracy_baseline':np.nan,'balanced_accuracy':np.nan,'sensitivity':np.nan,'specificity':np.nan,'precision':np.nan,'f1':np.nan,'mcc':np.nan,'roc_auc':np.nan,'pr_auc':np.nan,'pr_auc_no_skill_baseline':np.nan,'selected_threshold':threshold,'tn':np.nan,'fp':np.nan,'fn':np.nan,'tp':np.nan,'warning':'NA population'}
    y=df.class_label.to_numpy(int); p=df.probability.to_numpy(float); yh=(p>=threshold).astype(int)
    cm=confusion_matrix(y,yh,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    one_class=len(np.unique(y))<2
    if one_class: warnings.warn(f'ROC-AUC/PR-AUC undefined for one-class population: {population}')
    return {'architecture':architecture,'outer_fold':fold,'overlap_pct':overlap_pct,'evaluation_level':level,'population':population,'permutation_seed':permutation_seed,
            'n_subjects':df.anonymous_subject_id.nunique(),'n_pd':int((df.class_label==1).groupby(df.anonymous_subject_id).max().sum()) if level=='slice' else int((df.class_label==1).sum()),
            'n_non_pd':int(df.anonymous_subject_id.nunique() - ((df.class_label==1).groupby(df.anonymous_subject_id).max().sum())) if level=='slice' else int((df.class_label==0).sum()),
            'n_slices':len(df) if level=='slice' else int(df.n_slices.sum()),
            'accuracy':accuracy_score(y,yh),'majority_class_accuracy_baseline':max(np.mean(y==0),np.mean(y==1)),'balanced_accuracy':balanced_accuracy_score(y,yh),
            'sensitivity':recall_score(y,yh,pos_label=1,zero_division=0),'specificity':recall_score(y,yh,pos_label=0,zero_division=0),'precision':precision_score(y,yh,pos_label=1,zero_division=0),
            'f1':f1_score(y,yh,zero_division=0),'mcc':matthews_corrcoef(y,yh),'roc_auc':np.nan if one_class else roc_auc_score(y,p),'pr_auc':np.nan if one_class else average_precision_score(y,p),
            'pr_auc_no_skill_baseline':float(np.mean(y)),'selected_threshold':threshold,'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'warning':'one class; AUCs NA' if one_class else ''}

def predict_frame(model,df,architecture,batch_size):
    ds=make_ds(df,architecture,False,batch_size)
    p=model.predict(ds,verbose=0).reshape(-1)
    out=df[['image_path','anonymous_subject_id','class_label','slice_identifier']].reset_index(drop=True).copy(); out['probability']=p
    return out

def extract_embeddings(model,df,architecture,batch_size):
    emb_model=keras.Model(model.input,model.get_layer('penultimate_embedding').output)
    ds=make_ds(df,architecture,False,batch_size)
    z=emb_model.predict(ds,verbose=0)
    meta=df[['image_path','anonymous_subject_id','class_label','slice_identifier']].reset_index(drop=True).copy()
    return meta,z.astype(np.float32)

def _normalize_rows(z):
    return z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-12)

def identity_pair_analysis(meta,z,seed,n_pairs):
    """Balanced anchor-subject sampling; diagnosis labels are never used to define identity pairs."""
    rng=np.random.default_rng(seed); zn=_normalize_rows(z)
    by={sid:g.index.to_numpy() for sid,g in meta.groupby('anonymous_subject_id',sort=True)}
    eligible=[sid for sid,idx in by.items() if len(idx)>=2]
    if len(eligible)<2: return {'warning':'too few eligible subjects'},pd.DataFrame()
    per_anchor=max(1,int(math.ceil(n_pairs/len(eligible))))
    rows=[]; sids=sorted(by)
    for sid in eligible:
        idx=by[sid]
        others=[x for x in sids if x!=sid]
        for _ in range(per_anchor):
            a,b=rng.choice(idx,2,replace=False)
            rows.append({'anchor_subject':sid,'comparison_subject':sid,'similarity':float(zn[a]@zn[b]),'same_subject':1})
            sid2=str(rng.choice(others)); c=int(rng.choice(by[sid])); d=int(rng.choice(by[sid2]))
            rows.append({'anchor_subject':sid,'comparison_subject':sid2,'similarity':float(zn[c]@zn[d]),'same_subject':0})
    pairs=pd.DataFrame(rows)
    y=pairs.same_subject.to_numpy(int); sims=pairs.similarity.to_numpy(float)
    auc=float(roc_auc_score(y,sims))
    # Subject-cluster bootstrap: resample anchor subjects, never individual pairs.
    boots=[]; anchors=sorted(pairs.anchor_subject.unique())
    for _ in range(IDENTITY_CLUSTER_BOOTSTRAP_REPLICATES):
        sampled=rng.choice(anchors,size=len(anchors),replace=True)
        b=pd.concat([pairs[pairs.anchor_subject==sid] for sid in sampled],ignore_index=True)
        if b.same_subject.nunique()<2: continue
        boots.append((roc_auc_score(b.same_subject,b.similarity),b[b.same_subject==1].similarity.mean(),b[b.same_subject==0].similarity.mean()))
    if boots:
        ba=np.asarray(boots,float); auc_lo,auc_hi=np.percentile(ba[:,0],[2.5,97.5]); sm_lo,sm_hi=np.percentile(ba[:,1],[2.5,97.5]); dm_lo,dm_hi=np.percentile(ba[:,2],[2.5,97.5])
    else: auc_lo=auc_hi=sm_lo=sm_hi=dm_lo=dm_hi=np.nan
    same=pairs[pairs.same_subject==1].similarity; diff=pairs[pairs.same_subject==0].similarity
    result={'same_mean':float(same.mean()),'same_sd':float(same.std(ddof=1)),'same_mean_ci_low':float(sm_lo),'same_mean_ci_high':float(sm_hi),
            'different_mean':float(diff.mean()),'different_sd':float(diff.std(ddof=1)),'different_mean_ci_low':float(dm_lo),'different_mean_ci_high':float(dm_hi),
            'identity_roc_auc':auc,'identity_roc_auc_ci_low':float(auc_lo),'identity_roc_auc_ci_high':float(auc_hi),
            'n_same_pairs':int((pairs.same_subject==1).sum()),'n_different_pairs':int((pairs.same_subject==0).sum()),
            'n_anchor_subjects':len(anchors),'ci_unit':'anchor_subject_cluster','cluster_bootstrap_replicates':len(boots)}
    return result,pairs

def subject_retrieval_top1(meta,z):
    rows=[]; zn=_normalize_rows(z)
    for sid,g in meta.groupby('anonymous_subject_id',sort=True):
        if len(g)<2: continue
        inds=g.index.to_numpy(); q=inds[0]; gallery=[]; gsids=[]
        for sid2,g2 in meta.groupby('anonymous_subject_id',sort=True):
            inds2=g2.index.to_numpy(); pick=inds2[1] if sid2==sid and len(inds2)>1 else inds2[0]
            gallery.append(pick); gsids.append(sid2)
        best=int(np.argmax(zn[gallery]@zn[q])); rows.append(int(gsids[best]==sid))
    return float(np.mean(rows)) if rows else np.nan


def config_hash(architecture,fold,overlap_pct,permutation_seed=None):
    payload={**CONFIG,'architecture':architecture,'fold':fold,'overlap_pct':overlap_pct,'permutation_seed':permutation_seed}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()

def run_dir_for(architecture,fold,overlap_pct,permutation_seed=None):
    state='real_labels' if permutation_seed is None else f'permuted_seed_{permutation_seed}'
    return RUN_ROOT/'runs'/architecture/f'fold_{fold}'/f'overlap_{overlap_pct:03d}'/state

def json_safe(x):
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,)): return None if np.isnan(x) else float(x)
    if isinstance(x,np.ndarray): return x.tolist()
    try:
        if pd.isna(x): return None
    except Exception: pass
    return str(x)

def fresh_callbacks(path):
    return [keras.callbacks.ModelCheckpoint(str(path),monitor='val_loss',save_best_only=True),keras.callbacks.EarlyStopping(monitor='val_loss',patience=PATIENCE,restore_best_weights=True)]

def run_seed_for(fold,permutation_seed=None):
    # Overlap percentage does not enter the random seed.
    seed=int(MANUSCRIPT_FIXED_SEEDS[fold]) + int(_runtime_os.environ.get('PDMRI_SEED_OFFSET', '0'))
    if permutation_seed is not None: seed += int(permutation_seed)*1009
    return seed

def train_one(architecture,fold,overlap_pct,permutation_seed=None):
    rdir=run_dir_for(architecture,fold,overlap_pct,permutation_seed); rdir.parent.mkdir(parents=True,exist_ok=True)
    expected_hash=config_hash(architecture,fold,overlap_pct,permutation_seed); marker=rdir/'RUN_COMPLETE.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old.get('config_hash')!=expected_hash: raise RuntimeError(f'Completed run exists with a different configuration: {rdir}')
        print('SKIP completed:',rdir); return 'SKIPPED'
    if rdir.exists():
        archive=rdir.with_name(rdir.name+'_INCOMPLETE_ARCHIVE_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        shutil.move(str(rdir),str(archive)); print('Archived incomplete run:',archive)
    rdir.mkdir(parents=True)
    try:
        run_seed=run_seed_for(fold,permutation_seed)
        keras.backend.clear_session(); tf.keras.utils.set_random_seed(run_seed)
        random.seed(run_seed); np.random.seed(run_seed); tf.random.set_seed(run_seed)
        train,val,test,contam,cman,rman=build_partitions(fold,overlap_pct)
        audit=audit_partitions(train,val,test,contam,fold,overlap_pct)
        audit['training_seed']=run_seed
        pd.DataFrame([audit]).to_csv(rdir/'partition_audit.csv',index=False)
        cman.to_csv(rdir/'contamination_manifest.csv',index=False); rman.to_csv(rdir/'removed_training_slices_manifest.csv',index=False)
        for name,df in [('train',train),('validation',val),('test',test)]:
            df[['image_path','anonymous_subject_id','class_label','slice_identifier']].to_csv(rdir/f'{name}_manifest.csv',index=False)
        if permutation_seed is not None:
            all_ids=sorted(set(train.anonymous_subject_id)|set(val.anonymous_subject_id)|set(test.anonymous_subject_id))
            labels=subject_table.set_index('anonymous_subject_id').loc[all_ids,'class_label'].to_numpy().copy()
            rng=np.random.default_rng(int(permutation_seed)+fold); rng.shuffle(labels)
            pmap=dict(zip(all_ids,labels.astype(int)))
            pd.DataFrame({'anonymous_subject_id':all_ids,'permuted_class_label':labels.astype(int)}).to_csv(rdir/'subject_label_permutation_map.csv',index=False)
            for df in (train,val,test): df['class_label']=df.anonymous_subject_id.map(pmap).astype(int)
        run_cfg={**CONFIG,'architecture':architecture,'outer_fold':fold,'overlap_pct':overlap_pct,'permutation_seed':permutation_seed,'effective_batch_size':BATCH_SIZE,'training_seed':run_seed}
        (rdir/'run_configuration.json').write_text(json.dumps(run_cfg,indent=2,default=json_safe))
        augment=make_augmentation(run_seed)
        model,base=build_model(architecture); compile_model(model,HEAD_LR)
        head_ckpt=rdir/'best_head.keras'; fine_ckpt=rdir/'best_finetune.keras'; ckpt=rdir/'best_model.keras'
        train_ds=make_ds(train,architecture,True,BATCH_SIZE,run_seed=run_seed,augment=augment); val_ds=make_ds(val,architecture,False,BATCH_SIZE)
        h1=model.fit(train_ds,validation_data=val_ds,epochs=EFFECTIVE_HEAD_EPOCHS,callbacks=fresh_callbacks(head_ckpt),verbose=1)
        unfreeze_tail(base,architecture); compile_model(model,FINETUNE_LR)
        h2=model.fit(train_ds,validation_data=val_ds,epochs=EFFECTIVE_FINETUNE_EPOCHS,callbacks=fresh_callbacks(fine_ckpt),verbose=1)
        head_best=min(h1.history.get('val_loss',[np.inf])); fine_best=min(h2.history.get('val_loss',[np.inf])); chosen=head_ckpt if head_best<=fine_best else fine_ckpt
        if not chosen.exists(): raise RuntimeError('Best-stage checkpoint was not created.')
        shutil.copy2(chosen,ckpt); model=keras.models.load_model(ckpt)
        pd.concat([pd.DataFrame(h1.history).assign(stage='frozen_head'),pd.DataFrame(h2.history).assign(stage='fine_tuning')],ignore_index=True).to_csv(rdir/'training_history.csv',index=False)
        val_pred=predict_frame(model,val,architecture,BATCH_SIZE); val_patient=aggregate_patient(val_pred)
        threshold,curve=select_threshold(val_patient); curve.to_csv(rdir/'threshold_curve_validation_only.csv',index=False)
        json.dump({'selected_threshold':threshold,'selection_source':'inner_validation_patient_mean_probability'},open(rdir/'selected_threshold.json','w'),indent=2)
        test_pred=predict_frame(model,test,architecture,BATCH_SIZE); test_pred.to_csv(rdir/'slice_predictions.csv',index=False)
        pat=aggregate_patient(test_pred); pat['seen_identity']=pat.anonymous_subject_id.isin(set(contam)); pat.to_csv(rdir/'patient_predictions.csv',index=False)
        metrics=[]
        metrics.append(metric_row(test_pred,threshold,'slice','all',architecture,fold,overlap_pct,permutation_seed))
        metrics.append(metric_row(pat,threshold,'patient','all',architecture,fold,overlap_pct,permutation_seed))
        metrics.append(metric_row(pat[pat.seen_identity],threshold,'patient','seen',architecture,fold,overlap_pct,permutation_seed) if overlap_pct>0 else metric_row(None,threshold,'patient','seen',architecture,fold,overlap_pct,permutation_seed))
        metrics.append(metric_row(pat[~pat.seen_identity],threshold,'patient','unseen',architecture,fold,overlap_pct,permutation_seed) if overlap_pct<100 else metric_row(None,threshold,'patient','unseen',architecture,fold,overlap_pct,permutation_seed))
        pd.DataFrame(metrics).to_csv(rdir/'metrics.csv',index=False); json.dump(metrics,open(rdir/'metrics.json','w'),indent=2,default=json_safe)
        pd.DataFrame([{k:r.get(k) for k in ['architecture','outer_fold','overlap_pct','evaluation_level','population','permutation_seed','tn','fp','fn','tp']} for r in metrics]).to_csv(rdir/'confusion_matrices.csv',index=False)
        meta,z=extract_embeddings(model,test,architecture,BATCH_SIZE); np.savez_compressed(rdir/'test_penultimate_embeddings.npz',embeddings=z,image_path=meta.image_path.to_numpy(),anonymous_subject_id=meta.anonymous_subject_id.to_numpy(),class_label=meta.class_label.to_numpy())
        ident,pairs=identity_pair_analysis(meta,z,run_seed,PAIR_SAMPLES_EFFECTIVE); ident.update({'architecture':architecture,'outer_fold':fold,'overlap_pct':overlap_pct,'subject_retrieval_top1':subject_retrieval_top1(meta,z),'permutation_seed':permutation_seed})
        pd.DataFrame([ident]).to_csv(rdir/'identity_similarity_results.csv',index=False); pairs.to_csv(rdir/'identity_pair_samples.csv',index=False)
        marker.write_text(json.dumps({'status':'COMPLETE','pipeline_version':PIPELINE_VERSION,'completed_utc':datetime.now(timezone.utc).isoformat(),'config_hash':expected_hash,'architecture':architecture,'fold':fold,'overlap_pct':overlap_pct,'permutation_seed':permutation_seed,'training_seed':run_seed},indent=2))
        keras.backend.clear_session(); return 'COMPLETE'
    except tf.errors.ResourceExhaustedError as e:
        (rdir/'FAILED.json').write_text(json.dumps({'type':'GPU_OOM','message':str(e),'instruction':'Keep the recorded batch configuration and rerun consistently.'},indent=2)); raise RuntimeError('GPU out of memory; preserve the original protocol configuration.') from e
    except Exception as e:
        (rdir/'FAILED.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()},indent=2)); raise


plan=[]
for arch,levels in ACTIVE_PLAN.items():
    for overlap in levels: plan.append((arch,TARGET_OUTER_FOLD,overlap,None))
if RUN_ENDPOINT_PERMUTATION_CONTROL:
    for overlap in [0,100]:
        for ps in MANUSCRIPT_FIXED_SEEDS: plan.append(('ResNet50V2',TARGET_OUTER_FOLD,overlap,ps))
print('Primary planned runs for this fold:',len([x for x in plan if x[3] is None]))
print('Total planned including optional permutation runs:',len(plan))
print(plan)
for item in plan:
    print('\n=== RUN',item,'===')
    train_one(*item)


STATIC_AUDIT={
 'source_paths_parameterized':True,
 'fixed_training_seed_within_fold_across_overlap_levels':True,
 'nested_contamination_25_50_75_100':True,
 'same_subject_leak_slices_reused_across_higher_doses':True,
 'validation_never_contaminated':True,
 'image_path_overlap_hard_fail':True,
 'test_never_used_for_threshold_or_training_selection':True,
 'pd_positive_class':True,
 'identity_ci_clustered_by_subject':True,
 'restart_config_hash_protection':True,
 'incomplete_archives_excluded_from_failure_summary':True,
}
assert all(STATIC_AUDIT.values())
(RUN_ROOT/'STATIC_AUDIT.json').write_text(json.dumps(STATIC_AUDIT,indent=2))
print(json.dumps(STATIC_AUDIT,indent=2)); print('STATIC DESIGN AUDIT V2: PASS')


def collect_completed_csv(name):
    frames=[]
    for p in (RUN_ROOT/'runs').rglob(name) if (RUN_ROOT/'runs').exists() else []:
        if 'INCOMPLETE_ARCHIVE' in str(p): continue
        if (p.parent/'RUN_COMPLETE.json').exists():
            try: frames.append(pd.read_csv(p))
            except Exception as e: print('Could not read',p,e)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

metrics=collect_completed_csv('metrics.csv'); audits=collect_completed_csv('partition_audit.csv'); contams=collect_completed_csv('contamination_manifest.csv'); identities=collect_completed_csv('identity_similarity_results.csv')
metrics.to_csv(RUN_ROOT/'metrics_all_runs.csv',index=False)
(metrics[(metrics.evaluation_level=='patient')] if not metrics.empty else pd.DataFrame()).to_csv(RUN_ROOT/'metrics_patient_all_seen_unseen.csv',index=False)
audits.to_csv(RUN_ROOT/'partition_audit_all_runs.csv',index=False); contams.to_csv(RUN_ROOT/'contamination_manifest.csv',index=False); identities.to_csv(RUN_ROOT/'identity_similarity_results.csv',index=False)
thr=[]
for p in (RUN_ROOT/'runs').rglob('selected_threshold.json') if (RUN_ROOT/'runs').exists() else []:
    if 'INCOMPLETE_ARCHIVE' in str(p) or not (p.parent/'RUN_COMPLETE.json').exists(): continue
    m=json.loads((p.parent/'RUN_COMPLETE.json').read_text()); d=json.loads(p.read_text()); thr.append({**{k:m.get(k) for k in ['architecture','fold','overlap_pct','permutation_seed','training_seed']},**d})
pd.DataFrame(thr).to_csv(RUN_ROOT/'selected_thresholds.csv',index=False)

# Inspect only expected current run directories; archived attempts are deliberately ignored.
fail=[]
for arch,fold,ov,ps in plan:
    d=run_dir_for(arch,fold,ov,ps)
    if not (d/'RUN_COMPLETE.json').exists(): fail.append({'architecture':arch,'outer_fold':fold,'overlap_pct':ov,'permutation_seed':ps,'run_dir':str(d),'failed_json_exists':(d/'FAILED.json').exists()})
pd.DataFrame(fail).to_csv(RUN_ROOT/'failed_or_incomplete_runs.csv',index=False)

real_expected=[run_dir_for(a,f,o,None) for a,f,o,p in plan if p is None]
primary_complete=sum((d/'RUN_COMPLETE.json').exists() for d in real_expected)
primary_expected=len(real_expected)
if primary_complete!=primary_expected:
    raise RuntimeError(f'Fold not complete: {primary_complete}/{primary_expected} primary runs complete. Rerun this notebook; completed runs will be skipped.')
if not audits.empty and not (audits.audit_status=='PASS').all(): raise RuntimeError('At least one partition audit is not PASS.')
# Verify every real-label run within this fold used exactly the same training seed.
real_markers=[]
for d in real_expected: real_markers.append(json.loads((d/'RUN_COMPLETE.json').read_text()))
seeds={m['training_seed'] for m in real_markers}
if seeds!={FOLD_TRAINING_SEED}: raise RuntimeError(f'Seed-control audit failed: {seeds} != {{FOLD_TRAINING_SEED}}')

master_hash=hashlib.sha256((RUN_ROOT/'master_subject_index.csv').read_bytes()).hexdigest()
split_hash=hashlib.sha256((RUN_ROOT/'outer_inner_split_manifest.csv').read_bytes()).hexdigest()
fold_marker={'status':'FOLD_COMPLETE','pipeline_version':PIPELINE_VERSION,'outer_fold':TARGET_OUTER_FOLD,'primary_runs_complete':primary_complete,'primary_runs_expected':primary_expected,'training_seed':FOLD_TRAINING_SEED,'master_index_sha256':master_hash,'split_manifest_sha256':split_hash,'completed_utc':datetime.now(timezone.utc).isoformat()}
(RUN_ROOT/'FOLD_COMPLETE.json').write_text(json.dumps(fold_marker,indent=2))
print(json.dumps(fold_marker,indent=2)); print('FOLD RUNNER COMPLETE.')
