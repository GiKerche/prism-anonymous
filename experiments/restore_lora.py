"""Reassemble a deposited numeric adapter into a safetensors file."""
import argparse
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True,choices=[26082701,26082702,26082703])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert not args.output.exists(),'Refusing to overwrite an existing adapter'
    arrays={}
    for path in sorted((ROOT/f'reproduction/lora/adapter_{args.seed}').glob('layer_*.npz')):
        with np.load(path,allow_pickle=False) as part:
            for name in part.files:
                assert name not in arrays
                arrays[name]=part[name]
    assert len(arrays)==496 and sum(a.size for a in arrays.values())==43278336
    from safetensors.numpy import save_file
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save_file(arrays,str(args.output))
    print('Reconstructed',len(arrays),'numeric adapter tensors.')

if __name__=='__main__':main()
