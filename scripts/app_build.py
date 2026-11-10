"""Assemble Korea monthly import panel (HS6 x partner) from Comtrade preview JSON files."""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, glob, collections
import numpy as np, pandas as pd
RAW = f'{SP_ROOT}/data_raw/comtrade'
OUT = f'{SP_ROOT}/work/app'
rows = []
for f in sorted(glob.glob(f'{RAW}/k410_*.json')):
    d = json.load(open(f))
    for r in d.get('data', []):
        rows.append({k: r.get(k) for k in ['period','cmdCode','partnerCode','partnerISO','partner2Code','customsCode','motCode','netWgt','qty','qtyUnitAbbr','primaryValue','isNetWgtEstimated','flowCode','reporterCode']})
df = pd.DataFrame(rows)
print(df.shape, df.flowCode.unique(), df.reporterCode.unique())
print(df.groupby(['partner2Code','customsCode','motCode']).size())
df = df[(df.partner2Code==0)&(df.customsCode=='C00')&(df.motCode==0)]
dup = df.duplicated(['period','cmdCode','partnerCode']).sum(); print('dups', dup)
df['period'] = pd.to_datetime(df.period.astype(str), format='%Y%m')
print('netWgt missing', df.netWgt.isna().sum(), 'zero', (df.netWgt==0).sum(), 'estimated', df.isNetWgtEstimated.sum())
df.to_csv(f'{OUT}/korea_imports_long.csv', index=False)
w = df[df.partnerCode!=0]
tot = w.groupby(['cmdCode','partnerCode','partnerISO']).agg(kg=('netWgt','sum'), usd=('primaryValue','sum')).reset_index()
tot['share_kg'] = tot.kg / tot.groupby('cmdCode').kg.transform('sum')
tot['share_usd'] = tot.usd / tot.groupby('cmdCode').usd.transform('sum')
for c, g in tot.groupby('cmdCode'):
    print(c, g.sort_values('share_kg', ascending=False).head(6)[['partnerISO','share_kg','share_usd']].round(3).to_dict('records'))
