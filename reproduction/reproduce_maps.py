"""Refit benign-anchor PCA/ridge candidates and reproduce selected maps."""
import json
from pathlib import Path
import numpy as np
from map_math import pca_train, projection, error
from reproduce_selection import load, read

ROOT=Path(__file__).resolve().parent

def main():
    protocol=read(ROOT/'protocol.json');cfg=protocol['map']
    count=0
    for key in ['receiver_m1','receiver_m3','receiver_m5']:
        base=ROOT/key
        matrices={side:np.concatenate([load(p) for p in sorted((base/'anchors').glob(side+'_*.npz'))]).astype(float)
                  for side in ['source','receiver']}
        ps=pca_train(matrices['source'],max(cfg['pca_rank_grid']))
        pr=pca_train(matrices['receiver'],max(cfg['pca_rank_grid']))
        xv,_=projection(matrices['source'],np.arange(1024,1536),ps)
        yv,yn=projection(matrices['receiver'],np.arange(1024,1536),pr)
        expected=read(base/'map_selection.json');candidates=[];weights={}
        for rank in cfg['pca_rank_grid']:
            x,y=ps['train'][:,:rank],pr['train'][:,:rank];gram=x.T@x
            for penalty in cfg['ridge_lambda_relative_grid']:
                absolute=float(penalty*np.trace(gram)/rank)
                w=np.linalg.solve(gram+absolute*np.eye(rank),x.T@y)
                mse=error(xv[:,:rank],yv[:,:rank],yn,w)
                target=next(r for r in expected['candidates'] if r['rank']==rank and r['lambda_relative']==penalty)
                assert np.isclose(mse,target['validation_normalized_mse'],atol=1e-9,rtol=1e-8)
                candidates.append(dict(rank=rank,lambda_relative=penalty,validation_normalized_mse=mse))
                weights[rank,penalty]=w;count+=1
        winner=min(candidates,key=lambda r:(r['validation_normalized_mse'],r['rank'],-r['lambda_relative']))
        rank=winner['rank'];penalty=winner['lambda_relative']
        assert (rank,penalty)==(expected['selected']['rank'],expected['selected']['lambda_relative'])
        source=load(ROOT/key.replace('receiver','source')/'directions.npz','native_pooled').astype(float)
        d=((source@ps['basis'][:,:rank])@weights[rank,penalty])@pr['basis'][:,:rank].T
        d=(d/np.linalg.norm(d)).astype('<f4');d/=np.linalg.norm(d.astype(float))
        expected_d=load(base/'directions.npz','transported_native')
        assert np.allclose(d,expected_d,rtol=1e-5,atol=1e-7)
        xt,_=projection(matrices['source'],np.arange(1536,2048),ps)
        yt,ytn=projection(matrices['receiver'],np.arange(1536,2048),pr)
        mse=error(xt[:,:rank],yt[:,:rank],ytn,weights[rank,penalty])
        paper=read(ROOT.parent/'paper/data/current_cycle_configuration.json')['receivers'][key[-2:]]
        assert np.isclose(mse,paper['benign_test_diagnostic_normalized_mse'],atol=1e-9)
        print(key,'all candidates, selected rank/penalty, transported direction and diagnostic error reproduced',flush=True)
    print(json.dumps(dict(status='PASS',map_candidates=count)))

if __name__=='__main__':main()
