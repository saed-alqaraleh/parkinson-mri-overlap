#!/usr/bin/env python3
"""Multi-worker content integrity audit for frozen image-record assignments.

Output pairs use pseudonymous tokens rather than original image paths.
Scan and score batch digests are written with an atomic-commit protocol.
Workers require distinct IDs and the same process count. Execution stages:
scan, plan, score, finalize. Interrupted batches may be resumed or rescued
only after the other worker has stopped the relevant stage.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, itertools, json, math, os, platform, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, __version__ as pillow_version
try:
    import cv2
except ImportError:
    cv2 = None
    import scipy
    from scipy.fft import dctn

EXPECTED = {
 'master_subject_index_relative.csv':'fa751363f0d660b3b0a5fab0e7f85875c2bb8009ee8446456015ead4c679f044',
 'outer_inner_split_manifest.csv':'fc84c39c0a5ca28b1efeebf2ce4f01597dab53cd8c7e1517bd0ffe856ddd8271',
 'contamination_manifest_relative.csv':'eae2200a908ab0b932505a5ffb31043f74d501e79711f58c602ddb0558f8bd23',
 'removed_training_slices_manifest_relative.csv':'0d4bcc84d41cac2ef5746fa2e1bd949f0b2d6a042df35dd9a4bfba32c7207502',
}
SCHEMA_VERSION = 'PIXEL_AUDIT_V1_20261106'
SCAN_FIELDS=['record_id','status','byte_sha256','pixel_sha256','phash','shape','dtype','error']
CAND_FIELDS=['record_a','record_b','origin']
SCORE_FIELDS=['record_a','record_b','adjudication','phash_hamming','rmse128','corr128','subject_relationship']
SUMMARY_FIELDS=['outer_fold','overlap_pct','partition_a','partition_b','subject_relationship','eligible_file_a','eligible_file_b',
 'decoded_a','decoded_b','confirmed_exact_byte_pairs','confirmed_exact_pixel_pairs_additional','near_similarity_unresolved_pairs',
 'phash_candidate_nonconfirmed','affected_unique_files','affected_unique_subjects','near_candidate_status','exact_content_status','content_audit_status']
LOG_FIELDS=['outer_fold','overlap_pct','partition_a','partition_b','relationship','record_a','record_b','adjudication',
 'pixel_content_confirmed','phash_hamming','rmse128','corr128','volume_id_verified','manual_adjudication']

def now(): return datetime.now(timezone.utc).isoformat()
def sha(data): return hashlib.sha256(data).hexdigest()
def filehash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for buf in iter(lambda:f.read(2<<20),b''):h.update(buf)
    return h.hexdigest()
def token(rel):return 'I_'+sha(str(rel).encode())[:16]
def canon(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)
def atomic_bytes(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.partial_{os.getpid()}_{int(time.time()*1e6)}')
    try:
        with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        tmp.unlink(missing_ok=True)
def atomic_json(p,ob):atomic_bytes(p,(json.dumps(ob,indent=2,sort_keys=True)+'\n').encode())
def dump_csv_bytes(rows,fields):
    # In-memory batches are intentionally bounded.
    s=io.StringIO(newline='')
    w=csv.DictWriter(s,fieldnames=fields,lineterminator='\n');w.writeheader()
    for row in rows:w.writerow({k:row.get(k,'') for k in fields})
    return s.getvalue().encode('utf8')
def read_rows(p):
    with open(p,newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def codehash():return filehash(__file__)
def env():
    return {'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
     'pillow':pillow_version,'dct':'opencv:'+cv2.__version__ if cv2 is not None else 'scipy:'+scipy.__version__}
def job_signature():return sha(canon({'manifests':EXPECTED,'algorithm':SCHEMA_VERSION,'environment':env(),'codehash':codehash()}).encode())
def commit_batch(job,stage,number,rows,fields,signature,expected_key):
    folder=Path(job)/stage;folder.mkdir(parents=True,exist_ok=True)
    filename=folder/f'batch_{number:06d}.csv'
    data=dump_csv_bytes(rows,fields)
    atomic_bytes(filename,data)
    atomic_json(filename.with_suffix('.json'),{'schema':SCHEMA_VERSION,'signature':signature,'expected_key':expected_key,
      'rows':len(rows),'sha256':sha(data),'file':filename.name,'stage':stage,'committed_utc':now()})

def valid_batch(job,stage,number,fields,signature,expected_key,expected_count=None):
    filename=Path(job)/stage/f'batch_{number:06d}.csv'
    meta=filename.with_suffix('.json')
    if not filename.exists() or not meta.exists():return False
    try:
        j=json.loads(meta.read_text());f=filename.read_bytes()
        if not (j.get('signature')==signature and j.get('expected_key')==expected_key and j.get('sha256')==sha(f)
           and j.get('schema')==SCHEMA_VERSION and j.get('stage')==stage):return False
        with io.StringIO(f.decode('utf8')) as s:
            r=list(csv.DictReader(s))
        return bool(r or expected_count==0) and (not r or list(r[0])==fields) and (expected_count is None or len(r)==expected_count) and j['rows']==len(r)
    except Exception:return False

def stamp(job,phase,worker,**kw):
    atomic_json(Path(job)/f'{phase}_worker_{worker}.json',{'phase':phase,'worker':worker,'timestamp_utc':now(),**kw})
def required_inputs(args):
    md=Path(args.manifests)
    for fn,hashvalue in EXPECTED.items():
        if not (md/fn).is_file() or filehash(md/fn)!=hashvalue:raise RuntimeError('Frozen manifest missing/changed: '+fn)
    sys.path.insert(0,str(Path(args.code_root)))
    from src.manifests import load_frozen_manifests
    from src.audits import audit_reference_manifests,audit_all_reference_partitions
    m=load_frozen_manifests(md)
    audit=audit_reference_manifests(m)
    if audit['status']!='PASS':raise RuntimeError('Frozen manifest audit FAILED')
    if len(m.master)!=42147 or m.master.relative_path.nunique()!=42147:raise RuntimeError('Unexpected inventory size/uniqueness')
    parts=audit_all_reference_partitions(m)
    if len(parts)!=25 or not parts.audit_status.eq('PASS').all():raise RuntimeError('Partition audits failed')
    out=Path(args.job);out.mkdir(parents=True,exist_ok=True)
    p=out/'partition_reference_verification.csv'
    b=dump_csv_bytes(parts.drop(columns=['contaminated_subject_ids'],errors='ignore').to_dict('records'),list(parts.drop(columns=['contaminated_subject_ids'],errors='ignore').columns))
    if not p.exists():atomic_bytes(p,b)
    elif sha(b)!=filehash(p):raise RuntimeError('Existing partition verification differs; stop without overwriting')
    refs=[(token(rel),str(rel)) for rel in m.master.relative_path.astype(str)]
    if len({t for t,r in refs})!=len(refs):raise RuntimeError('Pseudonym collision')
    return m,refs

def pixel_sha_orig(im):
    # Crucial: json.dumps(header,sort_keys=True) (with default separators) from original audit.
    im.load()
    if im.mode in ('P','PA'):im=im.convert('RGBA')
    z=np.asarray(im)
    if z.size==0:raise ValueError('Empty decoded pixel array')
    header={'mode':im.mode,'shape':list(z.shape),'dtype':str(z.dtype)}
    h=hashlib.sha256(json.dumps(header,sort_keys=True).encode()+b'\0'+np.ascontiguousarray(z).tobytes())
    return h.hexdigest(),str(list(z.shape)),str(z.dtype)

def phash64(im):
    z=np.asarray(ImageOps.grayscale(im).resize((32,32),Image.Resampling.BICUBIC),dtype=np.float32)
    coef=cv2.dct(z) if cv2 is not None else dctn(z,norm='ortho')
    d=coef[:8,:8].reshape(-1);med=float(np.median(d[1:]))
    return sum(int(q>med)<<k for k,q in enumerate(d))

def scan_one(pair,data_root):
    record_id,rel=pair
    try:
        root=Path(data_root).resolve();p=(root/rel).resolve()
        if not p.is_relative_to(root):raise ValueError('PATH_ESCAPES_DATA_ROOT')
        encoded=p.read_bytes();byte_sha=sha(encoded)
        with Image.open(io.BytesIO(encoded)) as im:
            px,shape,dtype=pixel_sha_orig(im)
            ph=f'{phash64(im):016x}'
        return dict(record_id=record_id,status='OK',byte_sha256=byte_sha,pixel_sha256=px,phash=ph,shape=shape,dtype=dtype,error='')
    except Exception as e:
        return dict(record_id=record_id,status='FAIL',byte_sha256='',pixel_sha256='',phash='',shape='',dtype='',error=type(e).__name__)

def scan_chunk(chunk,data_root):return [scan_one(p,data_root) for p in chunk]

def chunks(xs,n):return [xs[i:i+n] for i in range(0,len(xs),n)]
def batches_meta(xs,n):
    for i,chunk in enumerate(chunks(xs,n)):
        yield i,chunk,sha(canon([x[0] if isinstance(x,tuple) else x for x in chunk]).encode())

def guarded_job_config(args):
    j=Path(args.job)/'job_config.json'
    cfg={'manifest_sha256':EXPECTED,'schema':SCHEMA_VERSION,'code_sha256':codehash(),'runtime':env(),
         'signature':job_signature(),'scan_batch_size':args.scan_batch,'score_batch_size':args.score_batch}
    if j.exists():
        if json.loads(j.read_text())!=cfg:raise RuntimeError('Job configuration mismatch. Use matching notebooks/dependencies or a NEW job folder.')
    else:atomic_json(j,cfg)
    return cfg

def scan_stage(args,refs):
    if not Path(args.data_root).exists():raise RuntimeError('DATA_ROOT missing: mount/link MRI tree first')
    sig=job_signature(); allmeta=list(batches_meta(refs,args.scan_batch))
    allowed=[(i,x,k) for i,x,k in allmeta if args.rescue or i%args.world_size==args.worker_id]
    already=sum(valid_batch(args.job,'scan',i,SCAN_FIELDS,sig,k,len(chunk)) for i,chunk,k in allowed)
    pending=[(i,chunk,k) for i,chunk,k in allowed if not valid_batch(args.job,'scan',i,SCAN_FIELDS,sig,k,len(chunk))]
    n_total=sum(len(chunk) for _,chunk,_ in allowed)
    done=sum(len(chunk) for i,chunk,k in allowed if valid_batch(args.job,'scan',i,SCAN_FIELDS,sig,k,len(chunk)))
    started=time.monotonic();failure=0
    workers=max(1,min(args.cpus,max(1,os.cpu_count() or 1),4))
    print(f'[SCAN] worker {args.worker_id}/{args.world_size}  CPU processes={workers}  total batches={len(allowed)} '
          f'reused={already} pending={len(pending)} saved files={done}/{n_total}',flush=True)
    stamp(args.job,'scan',args.worker_id,state='RUNNING',batches_reused=already,files_reused=done,files_total=n_total)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(scan_chunk,chunk,args.data_root):(i,chunk,k) for i,chunk,k in pending}
            for future in as_completed(futures):
                i,chunk,k=futures[future];rows=future.result()
                if [r['record_id'] for r in rows]!=[t for t,r in chunk]:raise RuntimeError('Scan batch row ID mismatch')
                commit_batch(args.job,'scan',i,rows,SCAN_FIELDS,sig,k)
                done+=len(rows);failure+=sum(x['status']!='OK' for x in rows)
                elapsed=max(0.01,time.monotonic()-started)
                newly=done-(n_total-sum(len(c) for _,c,_ in pending))
                rate=newly/elapsed;remaining=n_total-done
                print(f'[SCAN] batch={i} files={done}/{n_total} reused={n_total-sum(len(c) for _,c,_ in pending)} '
                     f'new_failures={failure} new_rate={rate:.2f}/s ETA={remaining/max(rate,1e-8)/60:.1f}m',flush=True)
                stamp(args.job,'scan',args.worker_id,state='RUNNING',batches_reused=already,files_done=done,
                      files_total=n_total,new_failures=failure,files_per_second=round(rate,3),eta_minutes=round(remaining/max(rate,1e-8)/60,2))
    except BaseException as e:
        stamp(args.job,'scan',args.worker_id,state='INTERRUPTED',error=type(e).__name__,files_done=done,files_total=n_total)
        raise
    stamp(args.job,'scan',args.worker_id,state='DONE',files_done=done,files_total=n_total,new_failures=failure)

def merged_inventory(args,refs):
    sig=job_signature();meta=list(batches_meta(refs,args.scan_batch));allrows=[]
    for i,chunk,k in meta:
        if not valid_batch(args.job,'scan',i,SCAN_FIELDS,sig,k,len(chunk)):
            raise RuntimeError(f'NEED_SCAN_BATCH_{i:06d}. Run other worker or --rescue only after its process stops.')
        rows=read_rows(Path(args.job)/'scan'/f'batch_{i:06d}.csv')
        if [r['record_id'] for r in rows]!=[x[0] for x in chunk]:raise RuntimeError('Inventory ID/order mismatch')
        allrows.extend(rows)
    if len(allrows)!=len(refs):raise RuntimeError('Inventory count mismatch')
    data=dump_csv_bytes(allrows,SCAN_FIELDS)
    out=Path(args.job)/'pixel_hash_inventory.csv'
    if out.exists() and filehash(out)!=sha(data):raise RuntimeError('Existing merged inventory changed')
    if not out.exists():atomic_bytes(out,data)
    fails=[{'record_id':r['record_id'],'error':r['error']} for r in allrows if r['status']!='OK']
    atomic_bytes(Path(args.job)/'pixel_decode_failures.csv',dump_csv_bytes(fails,['record_id','error']))
    print(f'[MERGE] {len(allrows):,} files, {len(fails)} read/decode failures, inventory SHA={sha(data)}',flush=True)
    return allrows

def record_candidates(h,max_hash_pairs,max_record_pairs):
    # Same complete radius-5 six-part pigeonhole construction as original audit.
    byph=defaultdict(list)
    for rid,p in h.items():byph[int(p,16)].append(rid)
    buckets=defaultdict(list)
    for val in byph:
        off=0
        for i,bits in enumerate((11,11,11,11,10,10)):
            buckets[(i,(val>>off)&((1<<bits)-1))].append(val);off+=bits
    hashes=set()
    for vals in buckets.values():
        if len(vals)>50000:raise RuntimeError('NEAR_BUCKET_CAP')
        for a,b in itertools.combinations(vals,2):
            if (a^b).bit_count()<=5:
                hashes.add((min(a,b),max(a,b)))
                if len(hashes)>max_hash_pairs:raise RuntimeError('NEAR_HASH_PAIR_CAP')
    candidates=set()
    for ids in byph.values():
        for a,b in itertools.combinations(ids,2):
            candidates.add((min(a,b),max(a,b)))
            if len(candidates)>max_record_pairs:raise RuntimeError('NEAR_RECORD_PAIR_CAP')
    for a,b in hashes:
        for x in byph[a]:
            for y in byph[b]:
                candidates.add((min(x,y),max(x,y)))
                if len(candidates)>max_record_pairs:raise RuntimeError('NEAR_RECORD_PAIR_CAP')
    return candidates

def candidate_plan(args,refs):
    allrows=merged_inventory(args,refs)
    success={r['record_id']:r for r in allrows if r['status']=='OK'}
    exact=set()
    for field in ('byte_sha256','pixel_sha256'):
        by=defaultdict(list)
        for rid,r in success.items():by[r[field]].append(rid)
        for ids in by.values():
            if len(ids)*(len(ids)-1)//2>args.max_exact_pairs:
                raise RuntimeError('EXACT_PAIR_CAP: exact complete summary requires scalable cluster mode')
            for a,b in itertools.combinations(ids,2):
                exact.add((min(a,b),max(a,b)))
                if len(exact)>args.max_exact_pairs:raise RuntimeError('EXACT_PAIR_CAP')
    try:
        near=record_candidates({rid:r['phash'] for rid,r in success.items()},args.max_hash_pairs,args.max_record_pairs)
        near_status='COMPLETE_CANDIDATE_SEARCH'
    except RuntimeError as e:
        near=set();near_status='INCOMPLETE_NEAR_CANDIDATE_SEARCH:'+str(e)
    pairs=sorted(exact|near)
    # Candidate plan includes all globally identical-byte/pixel pairs even if near-search capped.
    plan_source=sha((filehash(Path(args.job)/'pixel_hash_inventory.csv')+job_signature()).encode())
    sig=job_signature()+':'+plan_source
    batches=[(i,chunk,sha(canon(chunk).encode())) for i,chunk in enumerate(chunks(pairs,args.score_batch))]
    for i,chunk,k in batches:
        fields=[{'record_a':a,'record_b':b,'origin':'exact' if (a,b) in exact else 'near'} for a,b in chunk]
        commit_batch(args.job,'candidate_plan',i,fields,CAND_FIELDS,sig,k)
        if i%10==0:print(f'[PLAN] saved {i+1}/{len(batches)} candidate batches',flush=True)
    plan={'schema':SCHEMA_VERSION,'signature':job_signature(),'plan_signature':sig,'plan_source':plan_source,
      'scan_sha256':filehash(Path(args.job)/'pixel_hash_inventory.csv'),
      'exact_pairs':len(exact),'near_pairs':len(near) if near_status=='COMPLETE_CANDIDATE_SEARCH' else None,
      'near_status':near_status,'total_candidate_pairs':len(pairs),'score_batches':len(batches),
      'score_batch_size':args.score_batch,'complete_utc':now()}
    atomic_json(Path(args.job)/'candidate_plan.json',plan)
    print(f'[PLAN] exact={len(exact):,} near={len(near):,} status={near_status} total={len(pairs):,}',flush=True)

def read_plan(args):
    p=Path(args.job)/'candidate_plan.json'
    if not p.exists():raise RuntimeError('Candidate plan missing; run --stage plan after both scan workers finish.')
    plan=json.loads(p.read_text())
    if plan['signature']!=job_signature() or plan['score_batch_size']!=args.score_batch:raise RuntimeError('Candidate plan/code mismatch')
    if plan['scan_sha256']!=filehash(Path(args.job)/'pixel_hash_inventory.csv'):raise RuntimeError('Scan inventory changed since plan')
    return plan

def grey(p):
    with Image.open(p) as im:
        return np.asarray(ImageOps.grayscale(im).resize((128,128),Image.Resampling.BICUBIC),dtype=np.float32)/255.

def score_chunk(rows,by_id,meta,data_root):
    root=Path(data_root).resolve()
    cache={}
    def preview(t):
        if t not in cache:
            path=(root/by_id[t]).resolve()
            if not path.is_relative_to(root):raise RuntimeError('PATH_ESCAPES_DATA_ROOT')
            cache[t]=grey(path)
        return cache[t]
    out=[]
    for pair in rows:
        a,b=pair['record_a'],pair['record_b'];ra,rb=meta[a],meta[b]
        relation='same_subject' if ra['subject']==rb['subject'] else 'different_subject'
        if ra['byte_sha256']==rb['byte_sha256']:decision='CONFIRMED_EXACT_BYTES';rmse='';corr=''
        elif ra['pixel_sha256']==rb['pixel_sha256']:decision='CONFIRMED_EXACT_DECODED_PIXELS';rmse='';corr=''
        elif pair['origin']=='near':
            try:
                x,y=preview(a),preview(b)
                rmse=float(np.sqrt(np.mean(np.square(x-y))))
                corr=float(np.corrcoef(x.ravel(),y.ravel())[0,1]) if x.std()>1e-8 and y.std()>1e-8 else float('nan')
                decision='HIGH_SIMILARITY_NEEDS_SOURCE_REVIEW' if rmse<=.02 and corr>=.995 else 'PHASH_CANDIDATE_NOT_CONFIRMED'
            except Exception:
                rmse='';corr='';decision='SIMILARITY_SCORING_FAILED'
        else:raise RuntimeError('Unknown candidate origin')
        out.append({'record_a':a,'record_b':b,'adjudication':decision,
          'phash_hamming':(int(ra['phash'],16)^int(rb['phash'],16)).bit_count(),
          'rmse128':rmse,'corr128':corr,'subject_relationship':relation})
    return out

def score_stage(args,m,refs):
    plan=read_plan(args)
    inventory=read_rows(Path(args.job)/'pixel_hash_inventory.csv')
    bymeta=m.master[['relative_path','anonymous_subject_id']]
    by_id={token(x):str(x) for x in bymeta.relative_path}
    subject_index=bymeta.set_index('relative_path')['anonymous_subject_id'].astype(str).to_dict()
    meta={r['record_id']:{**r,'subject':subject_index[by_id[r['record_id']]]} for r in inventory if r['status']=='OK'}
    sig=plan['plan_signature']
    ids=list(range(plan['score_batches']))
    allowed=[i for i in ids if args.rescue or i%args.world_size==args.worker_id]
    valid=[];pending=[]
    for i in allowed:
        cand=Path(args.job)/'candidate_plan'/f'batch_{i:06d}.csv'
        rows=read_rows(cand)
        key=sha(canon([(x['record_a'],x['record_b']) for x in rows]).encode())
        if not valid_batch(args.job,'candidate_plan',i,CAND_FIELDS,sig,key,len(rows)):
            raise RuntimeError(f'Candidate plan batch {i} failed validation')
        if valid_batch(args.job,'score',i,SCORE_FIELDS,sig,key,len(rows)):valid.append((i,len(rows)))
        else:pending.append((i,rows,key))
    done=sum(n for i,n in valid);total=done+sum(len(rows) for i,rows,k in pending)
    reused=done;start=time.monotonic();workers=max(1,min(args.cpus,max(1,os.cpu_count() or 1),4))
    print(f'[SCORE] worker={args.worker_id}/{args.world_size} CPU processes={workers} reused={len(valid)} pending={len(pending)} scored={done}/{total}',flush=True)
    stamp(args.job,'score',args.worker_id,state='RUNNING',pairs_done=done,pairs_total=total)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(score_chunk,rows,by_id,meta,args.data_root):(i,rows,key) for i,rows,key in pending}
            for fut in as_completed(futures):
                i,rows,key=futures[fut];scored=fut.result()
                assert len(scored)==len(rows)
                commit_batch(args.job,'score',i,scored,SCORE_FIELDS,sig,key)
                done+=len(rows);rate=(done-reused)/max(.01,time.monotonic()-start)
                print(f'[SCORE] batch={i} pairs={done}/{total} reused={reused} new_rate={rate:.2f}/s ETA={(total-done)/max(rate,1e-8)/60:.1f}m',flush=True)
                stamp(args.job,'score',args.worker_id,state='RUNNING',pairs_done=done,pairs_total=total,
                    pairs_per_second=round(rate,2),eta_minutes=round((total-done)/max(rate,1e-8)/60,2))
    except BaseException as e:
        stamp(args.job,'score',args.worker_id,state='INTERRUPTED',pairs_done=done,pairs_total=total,error=type(e).__name__)
        raise
    stamp(args.job,'score',args.worker_id,state='DONE',pairs_done=done,pairs_total=total)

def final_stage(args,m,refs):
    plan=read_plan(args)
    inv=read_rows(Path(args.job)/'pixel_hash_inventory.csv')
    succeeded={r['record_id'] for r in inv if r['status']=='OK'}
    failing=len(inv)-len(succeeded)
    out=Path(args.job)
    meta=m.master.assign(record_id=m.master.relative_path.map(token)).set_index('record_id')
    subjects=meta.anonymous_subject_id.astype(str).to_dict()
    memberships={};totals={}
    from src.manifests import build_partitions
    for fold in range(5):
        for ov in (0,25,50,75,100):
            tr,va,te,_=build_partitions(m,fold,ov)
            roles={}
            sets={p:{token(s) for s in z.relative_path.astype(str)} for p,z in [('train',tr),('validation',va),('test',te)]}
            for p,rs in sets.items():
                if any(x in roles for x in rs):raise RuntimeError('Duplicate role for same image in partition')
                roles.update({x:p for x in rs})
            memberships[(fold,ov)]=roles
            totals[(fold,ov)]={p:(len(rs),len(rs&succeeded)) for p,rs in sets.items()}
    pairs=[('train','test'),('train','validation'),('validation','test')]
    counters=defaultdict(Counter);files=defaultdict(set);subjs=defaultdict(set)
    log_path=out/'duplicate_candidate_adjudication_log.csv'
    log_tmp=log_path.with_name(log_path.name+f'.partial_{os.getpid()}')
    evaluated=0;scoring_fail=0;logrows=0
    with log_tmp.open('w',newline='',encoding='utf8') as lf:
        writer=csv.DictWriter(lf,fieldnames=LOG_FIELDS,lineterminator='\n');writer.writeheader()
        for i in range(plan['score_batches']):
            cand=read_rows(out/'candidate_plan'/f'batch_{i:06d}.csv')
            key=sha(canon([(x['record_a'],x['record_b']) for x in cand]).encode())
            if not valid_batch(out,'score',i,SCORE_FIELDS,plan['plan_signature'],key,len(cand)):
                raise RuntimeError(f'MISSING_VALID_SCORE_BATCH_{i:06d}')
            scored=read_rows(out/'score'/f'batch_{i:06d}.csv')
            for s in scored:
                a,b=s['record_a'],s['record_b'];decision=s['adjudication']
                if decision=='SIMILARITY_SCORING_FAILED':scoring_fail+=1
                relation=s['subject_relationship']
                for (fold,ov),role in memberships.items():
                    pa,pb=role.get(a),role.get(b)
                    if pa is None or pb is None or pa==pb:continue
                    if (pa,pb) not in pairs and (pb,pa) not in pairs:continue
                    if (pa,pb) not in pairs:pa,pb=pb,pa
                    key0=(fold,ov,pa,pb,relation)
                    counters[key0][decision]+=1
                    files[key0].update((a,b))
                    subjs[key0].update((subjects[a],subjects[b]))
                    writer.writerow({'outer_fold':fold,'overlap_pct':ov,'partition_a':pa,'partition_b':pb,
                        'relationship':relation,'record_a':a,'record_b':b,'adjudication':decision,
                        'pixel_content_confirmed':str(decision.startswith('CONFIRMED_')),
                        'phash_hamming':s['phash_hamming'],'rmse128':s['rmse128'],'corr128':s['corr128'],
                        'volume_id_verified':False,'manual_adjudication':'NOT_PERFORMED'})
                    logrows+=1
                evaluated+=1
            if (i+1)%10==0 or i+1==plan['score_batches']:
                print(f'[FINALIZE] read {i+1}/{plan["score_batches"]} scoring batches; {evaluated:,} candidate pairs; {logrows:,} assigned log rows',flush=True)
        lf.flush();os.fsync(lf.fileno())
    os.replace(log_tmp,log_path)
    near_complete=plan['near_status']=='COMPLETE_CANDIDATE_SEARCH' and scoring_fail==0
    exact_complete=failing==0
    summary=[]
    for fold in range(5):
        for ov in (0,25,50,75,100):
            for pa,pb in pairs:
                for rel in ['same_subject','different_subject']:
                    key0=(fold,ov,pa,pb,rel);c=counters[key0]
                    summary.append({'outer_fold':fold,'overlap_pct':ov,'partition_a':pa,'partition_b':pb,'subject_relationship':rel,
                        'eligible_file_a':totals[(fold,ov)][pa][0],'eligible_file_b':totals[(fold,ov)][pb][0],
                        'decoded_a':totals[(fold,ov)][pa][1],'decoded_b':totals[(fold,ov)][pb][1],
                        'confirmed_exact_byte_pairs':c['CONFIRMED_EXACT_BYTES'],
                        'confirmed_exact_pixel_pairs_additional':c['CONFIRMED_EXACT_DECODED_PIXELS'],
                        'near_similarity_unresolved_pairs':c['HIGH_SIMILARITY_NEEDS_SOURCE_REVIEW'] if near_complete else '',
                        'phash_candidate_nonconfirmed':c['PHASH_CANDIDATE_NOT_CONFIRMED'] if near_complete else '',
                        'affected_unique_files':len(files[key0]),'affected_unique_subjects':len(subjs[key0]),
                        'near_candidate_status':plan['near_status'],'exact_content_status':'COMPLETE' if exact_complete else 'PARTIAL',
                        'content_audit_status':'COMPLETE' if exact_complete and near_complete else 'PARTIAL'})
    assert len(summary)==150
    atomic_bytes(out/'duplicate_audit_summary.csv',dump_csv_bytes(summary,SUMMARY_FIELDS))
    atomic_bytes(out/'duplicate_audit_endpoints.csv',dump_csv_bytes([r for r in summary if r['overlap_pct'] in (0,100)],SUMMARY_FIELDS))
    record={'eligible_images':len(refs),'decoded_images':len(succeeded),'decode_failures':failing,
     'near_candidate_status':plan['near_status'],'exact_candidate_pairs':plan['exact_pairs'],
     'near_candidate_pairs_enumerated':plan['near_pairs'],'scored_candidate_pairs':evaluated,
     'similarity_scoring_failures':scoring_fail,'audited_partition_cells':len(summary),
     'endpoint_partition_cells':60,'all_overlap_partition_cells':150,
     'manual_review_performed':False,'scan_volume_identity_verified':False,
     'phash_hamming_radius':5,'near_confirmation_rmse_max':.02,'near_confirmation_corr_min':.995,
     'manifest_sha256':EXPECTED,'runtime':env(),'job_signature':job_signature(),
     'source_inventory_sha256':plan['scan_sha256'],'candidate_plan_sha256':filehash(out/'candidate_plan.json'),
     'summary_sha256':filehash(out/'duplicate_audit_summary.csv'),
     'adjudication_log_sha256':filehash(log_path),'created_utc':now(),
     'status':'COMPLETE' if exact_complete and near_complete else 'PARTIAL'}
    atomic_json(out/'pixel_audit_execution_record.json',record)
    print('[FINALIZE] '+json.dumps(record,indent=2),flush=True)

def audit_status(args,refs):
    print('JOB:',args.job)
    sig=job_signature();total=math.ceil(len(refs)/args.scan_batch)
    scan_ok=0;donefiles=0
    for i,chunk,k in batches_meta(refs,args.scan_batch):
        if valid_batch(args.job,'scan',i,SCAN_FIELDS,sig,k,len(chunk)):
            scan_ok+=1;donefiles+=len(chunk)
    print(f'[STATUS] scan batches {scan_ok}/{total}; images checkpointed {donefiles}/{len(refs)}')
    plan_path=Path(args.job)/'candidate_plan.json'
    if plan_path.exists():
        plan=read_plan(args);good=0
        for i in range(plan['score_batches']):
            batch=read_rows(Path(args.job)/'candidate_plan'/f'batch_{i:06d}.csv')
            key=sha(canon([(x['record_a'],x['record_b']) for x in batch]).encode())
            if valid_batch(args.job,'score',i,SCORE_FIELDS,plan['plan_signature'],key,len(batch)):good+=1
        print(f'[STATUS] score batches {good}/{plan["score_batches"]}; near={plan["near_status"]}')
    for f in ['pixel_audit_execution_record.json','duplicate_audit_summary.csv']:
        print(f'[STATUS] {f}:', (Path(args.job)/f).exists())

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifests',required=True);ap.add_argument('--data-root',required=True)
    ap.add_argument('--code-root',required=True);ap.add_argument('--job',required=True)
    ap.add_argument('--stage',choices=['scan','plan','score','finalize','status'],required=True)
    ap.add_argument('--worker-id',type=int,default=0);ap.add_argument('--world-size',type=int,choices=(1,2),default=1)
    ap.add_argument('--cpus',type=int,default=2);ap.add_argument('--rescue',action='store_true')
    ap.add_argument('--scan-batch',type=int,default=256);ap.add_argument('--score-batch',type=int,default=10000)
    ap.add_argument('--max-hash-pairs',type=int,default=1500000)
    ap.add_argument('--max-record-pairs',type=int,default=1000000)
    ap.add_argument('--max-exact-pairs',type=int,default=2000000)
    args=ap.parse_args()
    if not (0<=args.worker_id<args.world_size):ap.error('worker-id must be less than world-size')
    if args.scan_batch<1 or args.score_batch<1:ap.error('batch size must be > 0')
    _,refs=required_inputs(args) if args.stage in ('scan','plan','status') else (None,None)
    # Only stages that need original dataset path require it. Others read reference manifests again.
    if args.stage in ('score','finalize'):
        m,refs=required_inputs(args)
    else:
        from src.manifests import load_frozen_manifests
        m=load_frozen_manifests(args.manifests)
    guarded_job_config(args)
    if args.stage=='scan':scan_stage(args,refs)
    elif args.stage=='plan':candidate_plan(args,refs)
    elif args.stage=='score':score_stage(args,m,refs)
    elif args.stage=='finalize':final_stage(args,m,refs)
    else:audit_status(args,refs)

if __name__=='__main__':main()
