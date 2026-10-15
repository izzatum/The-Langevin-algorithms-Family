"""Optional fixture regeneration; requires ArviZ 0.22, NumPy and SciPy, never test dependencies."""
import json, os
from pathlib import Path
os.environ['MPLCONFIGDIR']='/tmp/lafa-arviz-cache'
import numpy as np
import arviz as az
from arviz.stats.diagnostics import _ess_bulk, _ess_tail, _ess_mean, _ess_quantile
from scipy.stats import beta
rng=np.random.default_rng(137)
sets={}
sets['iid']=rng.normal(size=(4,128,2))
noise=rng.normal(size=(4,640,2)); ar=noise.copy()
for t in range(1,640): ar[:,t]=.85*ar[:,t-1]+noise[:,t]
sets['ar1']=ar[:,128:]
sets['ties']=np.round(rng.normal(size=(4,127,2)))
sets['heavy_tail']=rng.standard_t(1.5,size=(4,128,2))
sets['antithetic']=np.tile([1.,-1.],64)[None,:,None]*np.ones((4,1,2))+rng.normal(scale=.01,size=(4,128,2))
sets['shifted']=rng.normal(size=(4,128,2))+np.arange(4)[:,None,None]*2
out={'reference': 'ArviZ 0.22.0; numpy 2.3.5; scipy 1.17.0; generated with tests/data/generate_chain_quality_reference.py; capped split ESS at retained draw count; MCSE recalculated with capped ESS', 'cases':{}}
for name,x in sets.items():
 cap=x.shape[0]*2*(x.shape[1]//2)
 vals={k:[] for k in ['bulk','tail','mean_mcse','quantile_mcse']}
 for d in range(2):
  y=x[:,:,d]
  vals['bulk'].append(min(_ess_bulk(y),cap))
  vals['tail'].append(min(_ess_tail(y),cap))
  vals['mean_mcse'].append(y.std(ddof=1)/np.sqrt(min(_ess_mean(y),cap)))
  quant=[]
  for p in [.01,.05,.5,.95,.99]:
   if (y <= np.quantile(y,p)).all():
    quant.append(None)  # intentional undefined-indicator policy; ArviZ instead returns N
    continue
   ess=min(_ess_quantile(y,p),cap)
   endpoints=beta.ppf([.1586553,.8413447],ess*p+1,ess*(1-p)+1)*y.size-1
   a=max(int(np.floor(endpoints[0])),0); b=min(int(np.ceil(endpoints[1])),y.size-1)
   ordered=np.sort(y.ravel());quant.append((ordered[b]-ordered[a])/2)
  vals['quantile_mcse'].append(quant)
 out['cases'][name]={'chains':x.tolist(),**vals}
with Path(__file__).with_name('chain_quality_arviz_0_22.json').open('w') as f: json.dump(out,f,separators=(',',':'))
