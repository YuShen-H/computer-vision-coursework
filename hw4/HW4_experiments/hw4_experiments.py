"""Controlled HW4 experiments. Training never opens official test files."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

import hw4_core as h

ROOT = Path(__file__).resolve().parent
GROUPS = ('A_clean', 'B_rotation', 'C_mixed')
ERROR_INDICES = [565, 716, 6844, 9074, 3731, 8384]


def source_hashes():
    return {p.name: h.file_digest(p) for p in [Path(__file__), ROOT/'hw4_core.py']}


def profile(group):
    return dict(probability=0.0 if group == 'A_clean' else 0.8,
                families=['rotation'] if group == 'B_rotation' else ['rotation','noise','blur','brightness'],
                rotation_max=30.0, noise_max=40.0, blur_min=0.3, blur_max=2.0, brightness_min=0.4)


def controls(baseline, seed, group):
    keys = ['batch_size','learning_rate','min_learning_rate','weight_decay','dropout_rate',
            'conv_dropout','patience','min_delta','optimizer','scheduler','early_stopping','train_eval']
    cfg = {k: baseline['config'][k] for k in keys}
    expected = dict(batch_size=256, learning_rate=0.001, min_learning_rate=1e-5,
                    weight_decay=0.001, dropout_rate=0.5, conv_dropout=0.2,
                    patience=6, min_delta=0.0, optimizer='AdamW', scheduler='CosineAnnealingLR',
                    early_stopping=True, train_eval=True)
    if cfg != expected:
        raise ValueError('Reference settings differ from the verified HW3 baseline. Review before proceeding.')
    cfg.update(epochs=60, seed=seed, group=group, task=4, architecture_version='cnn_v2',
               augmentation=group!='A_clean', augmentation_profile=profile(group), scheduler_T_max=59,
               checkpoint_selection='lowest clean validation loss', training_from_scratch=True,
               normalization='float32(pixel)/255.0', source_sha256=source_hashes(),
               torch_version=str(h.torch.__version__), numpy_version=str(h.np.__version__),
               opencv_version=str(h.cv2.__version__))
    return cfg


def write_confusion(cm, title, path):
    fig, ax = h.plt.subplots(figsize=(10,8))
    im = ax.imshow(cm, cmap='Blues')
    ax.set(xticks=range(10), yticks=range(10), xticklabels=h.CLASSES, yticklabels=h.CLASSES,
           xlabel='Predicted class', ylabel='True class', title=title)
    h.plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(10):
        for j in range(10):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,
                    color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(path,dpi=160);h.plt.close(fig)


def score(images, labels, checkpoint, folder, device, batch_size, noise_seed):
    """Each condition is generated identically for every checkpoint, including noise."""
    folder.mkdir(parents=True, exist_ok=False)
    ck=h.read_checkpoint(checkpoint)
    model=h.frozen_model(ck,device); before=h.state_digest(model)
    rows=[]; per_class=[]; pred={'y_true':labels}; matrices={}; manifest=[]
    for case in h.conditions():
        pixels=h.corrupt(images,case,noise_seed)
        manifest.append(dict(case, pixels_sha256=h.array_digest(pixels)))
        loss,yp=h.evaluate(model,pixels,labels,batch_size,device,case['id'])
        met,per,cm=h.metrics(labels,yp)
        rows.append(dict(condition=case['id'],family=case['family'],value=case['value'],loss=loss,
                         accuracy_pct=100*met['accuracy'],macro_f1=met['macro_f1']))
        per_class.extend(dict(condition=case['id'],**x) for x in per)
        pred[case['id']]=yp;matrices[case['id']]=cm
        h.np.savetxt(folder/(case['id']+'_confusion.csv'),cm,fmt='%d',delimiter=',')
        if case['id']=='clean':
            write_confusion(cm,ck['config'].get('group','HW3 reference')+' / clean',folder/'clean_confusion.png')
    if h.state_digest(model)!=before:
        raise RuntimeError('Model state changed during evaluation.')
    corrupted=[r for r in rows if r['family']!='clean']
    families={f:float(h.np.mean([r['accuracy_pct'] for r in corrupted if r['family']==f]))
              for f in ['rotation','noise','blur','brightness']}
    summary=dict(clean_accuracy_pct=rows[0]['accuracy_pct'],clean_macro_f1=rows[0]['macro_f1'],
                 mean_corrupted_15_pct=float(h.np.mean([r['accuracy_pct'] for r in corrupted])),
                 mean_corrupted_equal_families_pct=float(h.np.mean(list(families.values()))),
                 **{f+'_accuracy_pct':v for f,v in families.items()})
    h.write_csv(folder/'metrics.csv',rows);h.write_csv(folder/'per_class.csv',per_class)
    h.np.savez_compressed(folder/'predictions.npz',**pred)
    h.np.savez_compressed(folder/'confusion_matrices.npz',**matrices)
    h.write_json(folder/'summary.json',summary)
    h.write_json(folder/'manifest.json',dict(conditions=manifest,data_sha256=h.data_digest(images,labels),
                 checkpoint_sha256=h.file_digest(checkpoint),state_sha256=before,noise_seed=noise_seed))
    del model
    return summary


def collect_summary(root, seeds, phase):
    rows=[]
    for seed in seeds:
        seed_rows=[]
        for group in GROUPS:
            run=root/f'seed_{seed}'/group
            summary=json.loads((run/phase/'summary.json').read_text())
            training=json.loads((run/'training_summary.json').read_text())
            row=dict(seed=seed,group=group,**summary,best_epoch=training['best_epoch'],
                     stop_epoch=training['final_epoch'],training_seconds=training['training_seconds'])
            seed_rows.append(row)
        base=seed_rows[0]
        for row in seed_rows:
            row['clean_change_vs_A_pp']=row['clean_accuracy_pct']-base['clean_accuracy_pct']
            row['corrupted_gain_vs_A_pp']=row['mean_corrupted_15_pct']-base['mean_corrupted_15_pct']
            row['target_met']=row['corrupted_gain_vs_A_pp']>=3 and row['clean_change_vs_A_pp']>=-1
        rows.extend(seed_rows)
    h.write_csv(root/(phase+'_summary.csv'),rows)
    stats=[]
    for group in GROUPS:
        selected=[r for r in rows if r['group']==group]
        item=dict(group=group,n_runs=len(selected))
        for key in ['clean_accuracy_pct','mean_corrupted_15_pct','rotation_accuracy_pct',
                    'clean_change_vs_A_pp','corrupted_gain_vs_A_pp']:
            vals=[r[key] for r in selected]
            item[key+'_mean']=float(h.np.mean(vals))
            item[key+'_sample_std']=float(h.np.std(vals,ddof=1)) if len(vals)>1 else ''
        stats.append(item)
    h.write_csv(root/(phase+'_aggregate.csv'),stats)
    fig,axes=h.plt.subplots(2,2,figsize=(12,8))
    for ax,family in zip(axes.flat,['rotation','noise','blur','brightness']):
        for group in GROUPS:
            sets=[]
            for seed in seeds:
                p=root/f'seed_{seed}'/group/phase/'metrics.csv'
                with p.open(encoding='utf-8-sig',newline='') as f:
                    sets.append({r['condition']:r for r in csv.DictReader(f)})
            cases=[c for c in h.conditions() if c['family']==family]
            clean_x=1 if family=='brightness' else 0
            pts=[(clean_x,float(h.np.mean([float(s['clean']['accuracy_pct']) for s in sets])))]
            pts += [(c['value'],float(h.np.mean([float(s[c['id']]['accuracy_pct']) for s in sets]))) for c in cases]
            pts.sort(reverse=family=='brightness')
            ax.plot([p[0] for p in pts],[p[1] for p in pts],'o-',label=group)
        ax.set(title=family,xlabel={'rotation':'Degrees','noise':'Sigma (0-255)','blur':'Sigma (pixels)',
                                  'brightness':'Brightness factor'}[family],ylabel='Accuracy (%)',ylim=(0,100))
        if family=='brightness':ax.invert_xaxis()
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/(phase+'_comparison.png'),dpi=170);h.plt.close(fig)
    return rows


def train(args, device, baseline):
    data=h.load_training(args.data_dir,args.split_file,baseline['config']['split_sha256'])
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    for seed in args.seeds:
        for group in GROUPS:
            out=root/f'seed_{seed}'/group
            cfg=controls(baseline,seed,group)
            cfg.update(split_sha256=data[6],validation_sha256=h.data_digest(data[2],data[3]),
                       training_data_sha256=h.data_digest(data[0],data[1]),
                       expected_test_sha256=baseline['config']['test_sha256'],device=str(device),
                       eval_batch_size=args.eval_batch_size,evaluation_noise_seed=42,
                       reference_checkpoint_sha256=h.file_digest(args.baseline))
            if (out/'complete.json').is_file():
                done=json.loads((out/'complete.json').read_text())
                saved=json.loads((out/'config.json').read_text())
                if any(saved.get(k)!=v for k,v in cfg.items()) or done['checkpoint_sha256']!=h.file_digest(out/'cnn_best.pt'):
                    raise ValueError(f'Existing run differs from current settings: {out}. Choose a new --output-dir.')
                print('Already completed; skipping:',out,flush=True);continue
            if out.exists():
                raise ValueError(f'Incomplete run exists: {out}. Use a new --output-dir, or move that incomplete group folder before retrying.')
            out.mkdir(parents=True)
            for p in [Path(__file__),ROOT/'hw4_core.py']:shutil.copy2(p,out/p.name)
            h.np.savez_compressed(out/'split_indices.npz',train_indices=data[4],val_indices=data[5])
            print('\nTRAIN:',group,'seed',seed,flush=True)
            h.fit_augmented(data,cfg,out,device)
            score(data[2],data[3],out/'cnn_best.pt',out/'validation',device,args.eval_batch_size,42)
            h.write_json(out/'complete.json',dict(checkpoint_sha256=h.file_digest(out/'cnn_best.pt'),
                         source_sha256=source_hashes(),initial_model_state_sha256=cfg['initial_model_state_sha256']))
        hashes={json.loads((root/f'seed_{seed}'/g/'complete.json').read_text())['initial_model_state_sha256'] for g in GROUPS}
        if len(hashes)!=1:raise RuntimeError('Initial weights differ across A/B/C.')
        manifests=[json.loads((root/f'seed_{seed}'/g/'validation/manifest.json').read_text())['conditions'] for g in GROUPS]
        if any(m!=manifests[0] for m in manifests):raise RuntimeError('Validation pixels differ across groups.')
    collect_summary(root,args.seeds,'validation')
    print('\nFinished. No test files were opened. Review:',root/'validation_summary.csv',flush=True)
    print('After fixing settings, run: python hw4_experiments.py evaluate (use the same --output-dir and --seeds).')


def error_comparison(images, labels, seedroot):
    preds={}
    for group in GROUPS:
        with h.np.load(seedroot/group/'test/predictions.npz',allow_pickle=False) as z:preds[group]=z['clean'].copy()
    fig,axes=h.plt.subplots(2,3,figsize=(14,9));rows=[]
    for ax,i in zip(axes.flat,ERROR_INDICES):
        ax.imshow(images[i],cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.axis('off')
        text=f'Index {i} | True: {h.CLASSES[labels[i]]}'
        for group in GROUPS:text+=f'\n{group}: {h.CLASSES[preds[group][i]]}'
        ax.set_title(text,fontsize=10)
        rows.append(dict(index=i,true_class=h.CLASSES[labels[i]],**{g:h.CLASSES[preds[g][i]] for g in GROUPS}))
    fig.suptitle('Same six HW3 error cases / new A-B-C clean predictions',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(seedroot/'test_error_cases.png',dpi=170);h.plt.close(fig)
    h.write_csv(seedroot/'test_error_cases.csv',rows)
    # Also retain all new A errors and regressions; the six illustrations are not an unbiased sample.
    a=preds['A_clean'];all_rows=[]
    ids=h.np.flatnonzero((a!=labels)|((a==labels)&((preds['B_rotation']!=labels)|(preds['C_mixed']!=labels))))
    for i in ids:
        all_rows.append(dict(index=int(i),true_label=int(labels[i]),**{g:int(preds[g][i]) for g in GROUPS},
                             A_correct=bool(a[i]==labels[i])))
    if all_rows:h.write_csv(seedroot/'all_A_errors_and_new_regressions.csv',all_rows)


def evaluate(args, device, baseline):
    frozen=[]
    for seed in args.seeds:
        for group in GROUPS:
            run=args.output_dir/f'seed_{seed}'/group
            done=json.loads((run/'complete.json').read_text())
            ck=h.read_checkpoint(run/'cnn_best.pt');cfg=ck['config']
            if done['checkpoint_sha256']!=h.file_digest(run/'cnn_best.pt'):raise ValueError('Checkpoint changed.')
            if cfg['source_sha256']!=source_hashes():raise ValueError('Code changed since training. Restore the saved sources.')
            if cfg['expected_test_sha256']!=baseline['config']['test_sha256']:raise ValueError('Wrong HW3 reference.')
            if cfg['evaluation_noise_seed']!=42:raise ValueError('Unexpected noise seed.')
            if (run/'test').exists():raise ValueError(f'Test output exists: {run}/test. Existing results were preserved.')
            frozen.append(dict(seed=seed,group=group,checkpoint_sha256=done['checkpoint_sha256'],
                               validation_summary=json.loads((run/'validation/summary.json').read_text())))
    # Save the protocol and validation evidence before reading the test data.
    h.write_json(args.output_dir/'test_protocol.json',dict(frozen_runs=frozen,source_sha256=source_hashes(),
                 timestamp=datetime.now().isoformat(),conditions=h.conditions(),noise_seed=42,
                 target='C vs A: corrupted mean gain >=3pp and clean change >=-1pp',
                 note='Official test set was previously used in HW3 and HW4 diagnosis; not a newly untouched dataset.'))
    images,labels=h.load_test(args.data_dir)
    if h.data_digest(images,labels)!=baseline['config']['test_sha256']:raise ValueError('Test data hash differs from HW3.')
    for seed in args.seeds:
        manifests=[]
        for group in GROUPS:
            run=args.output_dir/f'seed_{seed}'/group
            score(images,labels,run/'cnn_best.pt',run/'test',device,args.eval_batch_size,42)
            manifests.append(json.loads((run/'test/manifest.json').read_text())['conditions'])
        if any(m!=manifests[0] for m in manifests):raise RuntimeError('Test pixels differ across groups.')
        error_comparison(images,labels,args.output_dir/f'seed_{seed}')
    # Original 40-epoch checkpoint is historical reference, excluded from A/B/C effect estimates.
    score(images,labels,args.baseline,args.output_dir/'HW3_reference_test',device,args.eval_batch_size,42)
    rows=collect_summary(args.output_dir,args.seeds,'test')
    for r in rows:print(r['seed'],r['group'],'clean',round(r['clean_accuracy_pct'],2),
                        'corrupted',round(r['mean_corrupted_15_pct'],2),'gain vs A',round(r['corrupted_gain_vs_A_pp'],2))
    print('Final results:',args.output_dir/'test_summary.csv')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['check','train','evaluate'])
    p.add_argument('--data-dir',type=Path,default=ROOT.parent/'data')
    p.add_argument('--split-file',type=Path,default=ROOT/'reference/split_indices.npz')
    p.add_argument('--baseline',type=Path,default=ROOT/'reference/cnn_best.pt')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/hw4_abc')
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--device',choices=['cuda','cpu','auto'],default='cuda')
    p.add_argument('--eval-batch-size',type=int,default=256)
    args=p.parse_args()
    if not args.seeds or len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0:p.error('Seeds must be unique nonnegative integers.')
    if args.eval_batch_size<1:p.error('eval-batch-size must be positive.')
    if args.device=='cuda' and not h.torch.cuda.is_available():p.error('CUDA unavailable. Activate your HW3 PyTorch environment, or explicitly use --device cpu.')
    device=h.torch.device('cuda' if args.device=='auto' and h.torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    if device.type=='cpu':h.torch.set_num_threads(min(4,h.torch.get_num_threads()))
    h.cv2.setNumThreads(1)
    baseline=h.read_checkpoint(args.baseline,augmentation=False)
    controls(baseline,args.seeds[0],GROUPS[0])
    print('Device:',device,'| torch:',h.torch.__version__,'| data:',args.data_dir,flush=True)
    if args.mode=='check':
        data=h.load_training(args.data_dir,args.split_file,baseline['config']['split_sha256'])
        print('PASS: training data, stratified split and baseline settings verified; 48000 / 12000.')
        print('Test-file existence (not read):')
        for n in ['t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte']:
            print(n, (args.data_dir/n).is_file() or (args.data_dir/(n+'.gz')).is_file())
    elif args.mode=='train':train(args,device,baseline)
    else:evaluate(args,device,baseline)


if __name__=='__main__':
    main()
