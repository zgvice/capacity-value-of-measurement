"""Reconstruct frozen annual inputs from their three original public workbooks."""
from common import *
import sys
from official_workbooks import read_numeric_xlsx,profile_keys,validate_sparse,reliability_metrics,_serial_day,_model_dates
from scipy.stats import rankdata
from concurrent.futures import ProcessPoolExecutor

def read_book(arg):
    path,kind=arg
    return pd.read_excel(path,engine='openpyxl') if kind=='sparse' else read_numeric_xlsx(path)

def run():
    raw=DATA/'pjm';frozen=DATA/'annual';old=read_json(frozen/'official_reconciliation.json');sources=[]
    for name,digest in old['source_hashes'].items():
        p=raw/name
        if sha(p)!=digest:raise AssertionError('Official source hash: '+name)
        sources.append({'file':str(p.relative_to(ROOT)),'sha256':digest,'bytes':p.stat().st_size})
    names=['27bra-hourly-load-scenarios.xlsx','27bra-info-for-loss-of-load-hours.xlsx','27bra-montecarlo-by-date.xlsx']
    log('从三份官方原工作簿独立重建年度输入')
    with ProcessPoolExecutor(max_workers=3) as pool:
        load_result,df,mc_result=list(pool.map(read_book,[(str(raw/n),'sparse' if i==1 else 'numeric') for i,n in enumerate(names)]))
    headers,load=load_result;keys,years=profile_keys(headers[1:]);hours=load[:,0].astype(int);profiles=load[:,1:].T
    if not np.array_equal(hours,np.arange(8760)):raise AssertionError('Incomplete official annual load')
    atoms=pd.read_csv(frozen/'official_atoms.csv.gz',float_precision='round_trip');error=validate_sparse(df,atoms,8760)
    if len(atoms)!=40300 or len(keys)!=403 or abs(atoms.p.sum()-1)>1e-12:raise AssertionError('System probability registry')
    lookup=atoms.set_index(['Year','Rotation','Replication']).atom_id.to_dict()
    df['atom_id']=[lookup[x] for x in map(tuple,df[['Year','Rotation','Replication']].to_numpy())]
    from scipy import sparse
    known=sparse.load_npz(frozen/'official_known_h.npz').tocsr()
    rebuilt=sparse.csr_matrix((-df.Margin,(df.atom_id,df.HourBeginningInYear)),shape=known.shape)
    delta=known-rebuilt;max_h=float(abs(delta.data).max()) if delta.nnz else 0.
    if max_h>1e-9:raise AssertionError(('Published shortage reconstruction',max_h))
    mch,mc=mc_result
    if mch[1:]!=list(range(100)):raise AssertionError('Actual replication header')
    weather=mc[:,0].astype(int);perf=mc[:,1:].astype(int);indices=np.searchsorted(weather,_serial_day(df.Date))
    mismatches=int(np.count_nonzero(perf[indices,df.Replication.to_numpy(int)]!=_serial_day(df.ActualPerfDay)))
    if mismatches:raise AssertionError('Published Monte Carlo performance date')
    archive=np.load(frozen/'exact_path_groups.npz');groups={name:archive[name] for name in archive.files};mapping=groups['atom_to_group'];max_group=0.
    if not np.array_equal(groups['profile_ids'][mapping],atoms.profile_id):raise AssertionError('Path grouping changed weather profile')
    for atom in range(40300):
        j=int(mapping[atom]);sl=slice(groups['ptr'][j],groups['ptr'][j+1]);rr=known.getrow(atom)
        if not np.array_equal(rr.indices,groups['hours'][sl]):raise AssertionError('Grouped shortage-hour identity')
        max_group=max(max_group,float(np.max(abs(rr.data-groups['values'][sl]))) if len(rr.data) else 0.)
    if max_group>1e-9:raise AssertionError('Grouped shortage-value identity')
    if not np.allclose(groups['weights'],np.bincount(mapping,weights=atoms.p,minlength=len(groups['weights'])),atol=1e-14,rtol=0):raise AssertionError('Grouped path probabilities')
    months=np.array([pd.DatetimeIndex(_model_dates(year,8760)).month for year,rotation in keys])
    seasons=(months%12)//3;pressure=np.empty_like(profiles)
    for sid in range(4):
        mask=seasons==sid;pressure[mask]=(rankdata(profiles[mask],method='average')-.5)/mask.sum()
    background=np.load(frozen/'background_arrays.npz');pressure_error=float(np.max(abs(pressure-background['pressure'])))
    if pressure_error>1e-12 or not np.array_equal(seasons,background['seasons']):raise AssertionError('Official load-pressure mapping')
    shape=pd.read_csv(frozen/'local_shape_parameters.csv');bands=np.where(pressure<=.5,0,np.where(pressure<=.9,1,2));g=np.empty_like(pressure)
    for sid,label in enumerate(['winter','spring','summer','fall']):
        for bid,band in enumerate(['low','middle','high']):
            rows=shape[(shape.variant=='all_numeric')&(shape.source=='committed')&np.isclose(shape.u,.5)&(shape.season==label)&(shape.band==band)]
            if len(rows)!=1:raise AssertionError('Frozen headroom-shape mapping')
            g[(seasons==sid)&(bands==bid)]=float(rows.g.iloc[0])
    g_error=float(np.max(abs(g-background['g_committed_median_allnumeric'])))
    if g_error>1e-12:raise AssertionError('Frozen headroom-shape values')
    metrics=reliability_metrics(df);scale=read_json(frozen/'official_eue_scale_explanation_v2.json')
    info={'status':'passed','sources':sources,'profiles':403,'complete_annual_atoms':40300,'hours_per_atom':8760,
        'max_shortage_reconstruction_error_MW':max_h,'max_grouped_shortage_error_MW':max_group,
        'performance_date_mismatches':mismatches,'pressure_array_max_error':pressure_error,'headroom_shape_max_error':g_error,
        'raw_model_metrics':metrics,'display_EUE_MWh':scale['report_display_scale']['EUE_MWh'],
        'raw_model_EUE_unchanged_by_publisher_display_conversion':True,'nonshortage_margin_is_declared_completion':True}
    write_json(AUDIT/'annual_source_check.json',info);log('官方工作簿、40300原场景、1788路径分组和背景形状核对通过')
    return info

if __name__=='__main__':run()
