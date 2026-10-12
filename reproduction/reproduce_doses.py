"""Reapply the frozen capability, quality and one-standard-error dose rule."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    data=json.loads((ROOT/'paper/data/development_curves.json').read_text())
    data.update(json.loads((ROOT/'reproduction/receiver_native_dose_curves.json').read_text()))
    count=0
    for key,report in data.items():
        for row in report['curve']:
            benign=report['benign'][row['condition']]['tool_required']
            quality=['malformed','truncated','parser_invalid','quality_alert']
            admissible=benign['official_valid']['delta_pp']>=-5-1e-9 and all(row[m]['delta_pp']<=5+1e-9 and benign[m]['delta_pp']<=5+1e-9 for m in quality)
            assert admissible==row['admissible'],(key,row['condition'],'admissibility')
        for name,expected in report['selected_alphas'].items():
            expected=expected['alpha'] if isinstance(expected,dict) else expected
            sign='plus' if name.endswith('plus') else 'minus'
            context='ethical' if sign=='plus' else 'neutral'
            branch=name.rsplit('_',1)[0]
            rows=[r for r in report['curve'] if r['source']==branch and r['context']==context and r['admissible']]
            useful=[r for r in rows if r['signed_gain_pp']>0]
            if not useful:chosen=0.
            else:
                best=max(useful,key=lambda r:(r['signed_gain_pp'],-abs(r['alpha'])))
                threshold=best['signed_gain_pp']-best['call_proxy']['delta_se_pp']
                eligible=[r for r in useful if r['signed_gain_pp']>=threshold-1e-12]
                chosen=min(eligible,key=lambda r:abs(r['alpha']))['alpha']
            assert np.isclose(chosen,expected),(key,name,chosen,expected)
            count+=1
        print(key,'dose choices reproduced',flush=True)
    print(json.dumps(dict(status='PASS',selected_doses=count)))

if __name__=='__main__':main()
