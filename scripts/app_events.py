"""Stream-level interruption events from the Korea import panel.
stream = (HS6 input, origin). Event in month t if the stream is active (trailing-12-month share of the
input's imports >= 5%) and its import value falls below 50% of its trailing-12-month median.
"""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
W = f'{SP_ROOT}/work/app'
df = pd.read_csv(f'{W}/korea_imports_long.csv', parse_dates=['period'])
df = df[df.partnerCode != 0]
NAMES = {392: 'Japan', 156: 'China', 842: 'USA', 490: 'Taiwan', 276: 'Germany', 702: 'Singapore', 634: 'Qatar',
         124: 'Canada', 826: 'United Kingdom', 56: 'Belgium', 643: 'Russia', 804: 'Ukraine', 251: 'France',
         704: 'Viet Nam', 699: 'India', 344: 'Hong Kong', 458: 'Malaysia'}
INPUT = {281111: 'Hydrogen fluoride', 370790: 'Photoresists etc.', 392099: 'Plastic film (incl. FPI)',
         811292: 'Ga/Ge/In/Nb/V metal', 282560: 'Germanium oxides', 381800: 'Silicon wafers',
         280461: 'Silicon >=99.99%', 281290: 'Non-metal halides (incl. NF3)', 282619: 'Fluorides (incl. WF6)',
         280429: 'Rare gases (He, Ne, Kr, Xe)'}
months = pd.date_range('2014-01-01', '2025-12-01', freq='MS')
usd = df.pivot(index='period', columns=['cmdCode', 'partnerCode'], values='primaryValue').reindex(months)
kg = df.pivot(index='period', columns=['cmdCode', 'partnerCode'], values='netWgt').reindex(months)
# months with a value record but no weight: impute weight from value at the stream's median unit value
has_rec = usd.notna()
uv = (usd / kg.where(kg > 0)).median()
kg_imp = kg.where(kg.notna(), usd / uv)
n_imp = int((kg.isna() & has_rec).sum().sum())
val = kg_imp.where(has_rec, 0.0).fillna(0.0)          # no record = no imports from that origin
share_base = usd.fillna(0.0)
tot = share_base.T.groupby(level=0).sum().T                # input totals (value) by month
share12 = {}
roll = val.rolling(12, min_periods=12)
# trailing (t-12..t-1) quantities
trail_sum = share_base.shift(1).rolling(12, min_periods=12).sum()
trail_tot = tot.shift(1).rolling(12, min_periods=12).sum()
trail_med = val.shift(1).rolling(12, min_periods=12).median()
streams = []
for (c, p) in val.columns:
    s_all = share_base[(c, p)].sum() / tot[c].sum()
    if s_all >= 0.05:
        streams.append((c, p))
mon = months[months >= '2015-01-01']
recs = []
for (c, p) in streams:
    sh = trail_sum[(c, p)] / trail_tot[c]
    active = (sh >= 0.05) & (trail_med[(c, p)] > 0)
    low = (val[(c, p)] < 0.5 * trail_med[(c, p)]) & active
    ev = low & ~low.shift(1, fill_value=False)        # onset: first month of an interruption run
    for t in mon:
        recs.append(dict(period=t, code=c, partner=p, origin=NAMES.get(p, str(p)), input=INPUT[c],
                         active=bool(active[t]), event=int(ev[t]) if active[t] else -1,
                         value=float(val.loc[t, (c, p)]), share12=float(sh[t])))
E = pd.DataFrame(recs)
E.to_csv(f'{W}/stream_events.csv', index=False)
summ = E[E.active].groupby(['origin', 'input']).agg(months=('event', 'size'), events=('event', 'sum')).reset_index()
summ['rate'] = summ.events / summ.months
print(summ.sort_values(['origin', 'input']).to_string())
print('streams', len(streams), 'origins', E.origin.nunique(), 'imputed weights', n_imp)
# co-occurrence by origin-month
A = E[E.active]
co = A.groupby(['origin', 'period']).event.sum().reset_index()
for o in ['Japan', 'China', 'USA']:
    x = co[co.origin == o]
    print(o, 'months', len(x), 'k-fold counts', x.event.value_counts().sort_index().to_dict())
    print('  top months', x.sort_values('event', ascending=False).head(8)[['period', 'event']].astype(str).values.tolist())
