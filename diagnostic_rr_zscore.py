import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from data_loader import load_all
from signals import compute_zscore_standard, compute_zscore_move_adj
from regime import compute_regime
from backtest import backtest
from config import BACKTEST_START, DEFAULT_PARAMS, ZSCORE_MIN_PERIODS
from scipy import stats as sp_stats
from scipy.stats import spearmanr

print('Loading data...')
D = load_all()
D = compute_regime(D)
rr_1m = D['rr_1m'].dropna()
move_s = D['move'].dropna()
etf_ret = D['etf_ret']
regime = D['regime']
print()
print('RR_1M: %d obs from %s to %s' % (len(rr_1m), rr_1m.index.min().strftime('%Y-%m-%d'), rr_1m.index.max().strftime('%Y-%m-%d')))
print('MOVE:  %d obs' % len(move_s))
print()

# ============================================================================
# SECTION 1: RAW RR_1M SIGNAL PROPERTIES
# ============================================================================
print('=' * 72)
print('SECTION 1: RAW RR_1M SIGNAL PROPERTIES')
print('=' * 72)
print()
print('--- Autocorrelation ---')
lags = [1, 5, 21, 63, 126, 252]
print('%6s  %10s' % ('Lag', 'Autocorr'))
print('-' * 20)
for lag in lags:
    ac = rr_1m.autocorr(lag=lag)
    print('%6d  %10.4f' % (lag, ac))

print()
print('--- ADF Stationarity Test ---')
try:
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(rr_1m.values, maxlag=21, autolag='AIC')
    print('  ADF Statistic : %.4f' % adf_result[0])
    print('  p-value       : %.6f' % adf_result[1])
    print('  Lags Used     : %d' % adf_result[2])
    print('  N Obs Used    : %d' % adf_result[3])
    print('  Critical Values:')
    for k, v in adf_result[4].items():
        print('    %5s: %.4f' % (k, v))
    if adf_result[1] < 0.05:
        print('  => STATIONARY at 5%% level')
    else:
        print('  => NON-STATIONARY at 5%% level (fail to reject unit root)')
except ImportError:
    print('  statsmodels not available -- skipping ADF test')

print()
print('--- Half-Life of Mean Reversion ---')
rr_lag = rr_1m.shift(1)
delta_rr = rr_1m - rr_lag
valid = pd.DataFrame({'y': delta_rr, 'x': rr_lag}).dropna()
xv = valid['x'].values
yv = valid['y'].values
Xm = np.column_stack([np.ones(len(xv)), xv])
beta_coef = np.linalg.lstsq(Xm, yv, rcond=None)[0]
slope = beta_coef[1]
half_life = np.nan
if slope < 0:
    half_life = -np.log(2) / slope
    print('  Slope (beta)  : %.6f' % slope)
    print('  Half-life     : %.1f days (%.1f months)' % (half_life, half_life / 21))
else:
    half_life = 99999
    print('  Slope (beta)  : %.6f' % slope)
    print('  => No mean reversion detected (positive slope)')

print()
print('--- Distribution Statistics ---')
print('  Mean           : %.4f' % rr_1m.mean())
print('  Std            : %.4f' % rr_1m.std())
print('  Median         : %.4f' % rr_1m.median())
print('  Min            : %.4f' % rr_1m.min())
print('  Max            : %.4f' % rr_1m.max())
print('  Skewness       : %.4f' % sp_stats.skew(rr_1m.values))
print('  Kurtosis (exc) : %.4f' % sp_stats.kurtosis(rr_1m.values))
jb_stat, jb_p = sp_stats.jarque_bera(rr_1m.values)
print('  Jarque-Bera    : stat=%.2f, p=%.6f' % (jb_stat, jb_p))
if jb_p < 0.05:
    print('  => NON-NORMAL distribution')
else:
    print('  => Consistent with normality')

# ============================================================================
# SECTION 2: ROLLING WINDOW SENSITIVITY FOR z_std
# ============================================================================
print()
print('=' * 72)
print('SECTION 2: ROLLING WINDOW SENSITIVITY FOR z_std')
print('=' * 72)
windows = [63, 126, 189, 250, 378, 504]
bt_params = dict(DEFAULT_PARAMS)
bt_params['tl_calm'] = -3.5
bt_params['ts_calm'] = 3.0
bt_params['tl_stress'] = -4.0
print()
print('Backtest params: tl_calm=%.1f, ts_calm=%.1f, tl_stress=%.1f' % (bt_params['tl_calm'], bt_params['ts_calm'], bt_params['tl_stress']))
print('Backtest start: %s' % BACKTEST_START)
print()
print('%8s  %8s  %8s  %8s  %8s  %8s  %8s' % ('Window', 'Sharpe', 'Excess', 'N_Events', 'MDD', 'Ann_Ret', 'Vol'))
print('-' * 72)
window_results = []
for w in windows:
    z_w = compute_zscore_standard(D['rr_1m'], window=w, min_periods=min(63, w))
    res = backtest(z_w, regime, etf_ret, params=bt_params, start=BACKTEST_START)
    m = res['metrics']
    print('%8d  %8.3f  %8.4f  %8d  %8.4f  %8.4f  %8.4f' % (w, m['sharpe'], m['excess'], m['n_events'], m['mdd'], m['ann_ret'], m['vol']))
    window_results.append({'window': w, 'sharpe': m['sharpe'], 'excess': m['excess'], 'n_events': m['n_events'], 'mdd': m['mdd'], 'ann_ret': m['ann_ret'], 'vol': m['vol']})
wdf = pd.DataFrame(window_results)
print()
print('Best window by Sharpe: %d' % wdf.loc[wdf['sharpe'].idxmax(), 'window'])
print('Best window by Excess: %d' % wdf.loc[wdf['excess'].idxmax(), 'window'])
print('Sharpe std across windows: %.4f' % wdf['sharpe'].std())

# ============================================================================
# SECTION 3: FORWARD RETURN PREDICTABILITY BY Z-SCORE QUINTILE
# ============================================================================
print()
print('=' * 72)
print('SECTION 3: FORWARD RETURN PREDICTABILITY BY Z-SCORE QUINTILE')
print('=' * 72)
z_std_250 = compute_zscore_standard(D['rr_1m'], window=250)
tlt_px = D['etf_px']['TLT']
horizons = [5, 21, 63]
fwd_rets = {}
for h in horizons:
    fwd_rets[h] = tlt_px.pct_change(h).shift(-h)
align_df = pd.DataFrame({'z_std': z_std_250})
for h in horizons:
    align_df['fwd_%dd' % h] = fwd_rets[h]
align_df = align_df.dropna()
print()
print('Using %d observations with valid z_std (250d) and forward TLT returns' % len(align_df))
print()
qlabels = ['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)']
align_df['quintile'] = pd.qcut(align_df['z_std'], 5, labels=qlabels)
print('Mean Forward TLT Returns by Z-Score Quintile (annualized %%):')
print()
print('%12s  %20s  %10s  %10s  %10s  %6s' % ('Quintile', 'Z_std range', 'Fwd_5d', 'Fwd_21d', 'Fwd_63d', 'N'))
print('-' * 78)
quintile_means = {}
for q in qlabels:
    mask = align_df['quintile'] == q
    sub = align_df[mask]
    z_lo = sub['z_std'].min()
    z_hi = sub['z_std'].max()
    means = []
    row_str = '%12s  %20s' % (q, '[%.2f, %.2f]' % (z_lo, z_hi))
    for h in horizons:
        mean_ret = sub['fwd_%dd' % h].mean() * (252.0 / h) * 100
        means.append(mean_ret)
        row_str += '  %9.2f%%' % mean_ret
    row_str += '  %6d' % len(sub)
    print(row_str)
    quintile_means[q] = means
print()
print('Monotonicity Check (Q1 < Q2 < ... < Q5):')
for i, h in enumerate(horizons):
    vals = [quintile_means[q][i] for q in qlabels]
    monotone = all(vals[j] <= vals[j + 1] for j in range(len(vals) - 1))
    spread = vals[-1] - vals[0]
    rho, pval = spearmanr([1, 2, 3, 4, 5], vals)
    mono_str = 'YES' if monotone else 'NO '
    print('  Fwd_%2dd: Monotone=%s  Q5-Q1 spread=%+7.2f%%  Spearman rho=%.3f (p=%.3f)' % (h, mono_str, spread, rho, pval))
print()
q1_5d = quintile_means['Q1(low)'][0]
q5_5d = quintile_means['Q5(high)'][0]
q1_21d = quintile_means['Q1(low)'][1]
q5_21d = quintile_means['Q5(high)'][1]
print('  Q1 (most neg z) 5d fwd: %+.2f%%,  Q5 (most pos z) 5d fwd: %+.2f%%' % (q1_5d, q5_5d))
print('  Q1 (most neg z) 21d fwd: %+.2f%%,  Q5 (most pos z) 21d fwd: %+.2f%%' % (q1_21d, q5_21d))

# ============================================================================
# SECTION 4: CORRELATION ANALYSIS
# ============================================================================
print()
print('=' * 72)
print('SECTION 4: CORRELATION ANALYSIS')
print('=' * 72)
z_move = compute_zscore_move_adj(D['rr_1m'], D['move'])
corr_df = pd.DataFrame({'RR_1M': D['rr_1m'], 'z_std': z_std_250, 'z_move': z_move, 'MOVE': D['move'], 'TLT_fwd_21d': tlt_px.pct_change(21).shift(-21)}).dropna()
print()
print('%d overlapping observations' % len(corr_df))
print()
r1 = corr_df['z_std'].corr(corr_df['z_move'])
r2 = corr_df['z_std'].corr(corr_df['MOVE'])
r3 = corr_df['RR_1M'].corr(corr_df['MOVE'])
r4 = corr_df['z_move'].corr(corr_df['MOVE'])
print('  Corr(z_std, z_move)     : %.4f' % r1)
print('  Corr(z_std, MOVE)       : %.4f' % r2)
print('  Corr(RR_1M, MOVE)       : %.4f' % r3)
print('  Corr(z_move, MOVE)      : %.4f' % r4)
r5 = corr_df['RR_1M'].corr(corr_df['TLT_fwd_21d'])
r6 = corr_df['z_std'].corr(corr_df['TLT_fwd_21d'])
r7 = corr_df['z_move'].corr(corr_df['TLT_fwd_21d'])
r8 = corr_df['MOVE'].corr(corr_df['TLT_fwd_21d'])
print()
print('  Corr(RR_1M,  TLT_fwd_21d) : %.4f' % r5)
print('  Corr(z_std,  TLT_fwd_21d) : %.4f' % r6)
print('  Corr(z_move, TLT_fwd_21d) : %.4f' % r7)
print('  Corr(MOVE,   TLT_fwd_21d) : %.4f' % r8)
print()
print('--- Partial Correlation: RR_1M -> TLT_fwd_21d, controlling for MOVE ---')
partial_corr = np.nan
try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    X_move = add_constant(corr_df['MOVE'].values)
    res_rr = OLS(corr_df['RR_1M'].values, X_move).fit()
    rr_resid = res_rr.resid
    res_tlt = OLS(corr_df['TLT_fwd_21d'].values, X_move).fit()
    tlt_resid = res_tlt.resid
    partial_corr = np.corrcoef(rr_resid, tlt_resid)[0, 1]
    print('  Partial corr(RR_1M, TLT_fwd_21d | MOVE) : %.4f' % partial_corr)
    print('  Raw corr(RR_1M, TLT_fwd_21d)            : %.4f' % r5)
    if abs(partial_corr) < abs(r5) * 0.7:
        print('  => MOVE partially explains the RR->TLT link')
    else:
        print('  => MOVE does NOT explain the RR->TLT link')
except ImportError:
    print('  statsmodels not available -- skipping partial correlation')
print()
print('--- Full Correlation Matrix ---')
corr_cols = ['RR_1M', 'z_std', 'z_move', 'MOVE', 'TLT_fwd_21d']
corr_matrix = corr_df[corr_cols].corr()
print(corr_matrix.round(4).to_string())

# ============================================================================
# SECTION 5: COMPARE z_std vs z_move vs z_raw
# ============================================================================
print()
print('=' * 72)
print('SECTION 5: COMPARE z_std vs z_move vs z_raw')
print('=' * 72)
z_std_bt = compute_zscore_standard(D['rr_1m'], window=250)
res_std = backtest(z_std_bt, regime, etf_ret, params=bt_params, start=BACKTEST_START)
m_std = res_std['metrics']
z_move_bt = compute_zscore_move_adj(D['rr_1m'], D['move'])
res_move = backtest(z_move_bt, regime, etf_ret, params=bt_params, start=BACKTEST_START)
m_move = res_move['metrics']
rr_mean_exp = D['rr_1m'].expanding(min_periods=63).mean()
rr_std_exp = D['rr_1m'].expanding(min_periods=63).std()
z_raw = (D['rr_1m'] - rr_mean_exp) / rr_std_exp.replace(0, np.nan)
z_raw.name = 'z_raw'
res_raw = backtest(z_raw, regime, etf_ret, params=bt_params, start=BACKTEST_START)
m_raw = res_raw['metrics']
rr_rank = D['rr_1m'].rolling(250, min_periods=63).rank(pct=True)
rr_rank.name = 'z_raw_rank'
rank_params = dict(DEFAULT_PARAMS)
rank_params['tl_calm'] = 0.10
rank_params['ts_calm'] = 0.90
rank_params['tl_stress'] = 0.05
res_rank = backtest(rr_rank, regime, etf_ret, params=rank_params, start=BACKTEST_START)
m_rank = res_rank['metrics']
p10 = D['rr_1m'].rolling(250, min_periods=63).quantile(0.10)
p90 = D['rr_1m'].rolling(250, min_periods=63).quantile(0.90)
p10_median = p10.dropna().median()
p90_median = p90.dropna().median()
print()
print('RR_1M rolling percentile reference (250d window):')
print('  10th percentile (median over time): %.4f' % p10_median)
print('  90th percentile (median over time): %.4f' % p90_median)
print()
print('%20s  %8s  %8s  %8s  %8s  %8s  %8s' % ('Signal', 'Sharpe', 'Excess', 'N_Events', 'MDD', 'Ann_Ret', 'Vol'))
print('-' * 80)
for name, m in [('z_std (250d)', m_std), ('z_move (250d)', m_move), ('z_raw (expanding)', m_raw), ('z_raw (rank-based)', m_rank)]:
    print('%20s  %8.3f  %8.4f  %8d  %8.4f  %8.4f  %8.4f' % (name, m['sharpe'], m['excess'], m['n_events'], m['mdd'], m['ann_ret'], m['vol']))
z_comp = pd.DataFrame({'z_std': z_std_bt, 'z_move': z_move_bt, 'z_raw': z_raw, 'z_rank': rr_rank}).dropna()
print()
print('--- Correlation between signal variants (%d obs) ---' % len(z_comp))
print(z_comp.corr().round(4).to_string())
print()
print('--- Z-Score Distribution Summary ---')
print('%12s  %8s  %8s  %8s  %8s  %8s  %8s  %8s  %8s' % ('Signal', 'Mean', 'Std', 'Skew', 'Kurt', 'Min', 'Max', 'pct<-3', 'pct>3'))
print('-' * 92)
for name, z in [('z_std', z_std_bt), ('z_move', z_move_bt), ('z_raw', z_raw)]:
    zv = z.dropna()
    pct_below = (zv < -3).mean() * 100
    pct_above = (zv > 3).mean() * 100
    print('%12s  %8.3f  %8.3f  %8.3f  %8.3f  %8.2f  %8.2f  %7.2f%%  %7.2f%%' % (name, zv.mean(), zv.std(), sp_stats.skew(zv.values), sp_stats.kurtosis(zv.values), zv.min(), zv.max(), pct_below, pct_above))

# ============================================================================
# SUMMARY
# ============================================================================
print()
print('=' * 72)
print('DIAGNOSTIC SUMMARY')
print('=' * 72)
if half_life < 99999:
    hl_str = '%.0f days = %.1f months' % (half_life, half_life / 21)
else:
    hl_str = 'N/A (no mean reversion)'
if not np.isnan(partial_corr):
    pc_str = '%.4f' % partial_corr
else:
    pc_str = 'N/A'
print()
print('Key Findings:')
print()
print('1. RR_1M Autocorrelation: lag-1 = %.3f (highly persistent)' % rr_1m.autocorr(lag=1))
print('   Half-life ~ %s' % hl_str)
print()
print('2. Rolling Window: Best Sharpe at window=%d' % wdf.loc[wdf['sharpe'].idxmax(), 'window'])
print('   (Sharpe=%.3f)' % wdf['sharpe'].max())
print('   Sensitivity: std of Sharpe across windows = %.4f' % wdf['sharpe'].std())
print()
print('3. Forward Return Predictability:')
q5q1_5 = quintile_means['Q5(high)'][0] - quintile_means['Q1(low)'][0]
q5q1_21 = quintile_means['Q5(high)'][1] - quintile_means['Q1(low)'][1]
q5q1_63 = quintile_means['Q5(high)'][2] - quintile_means['Q1(low)'][2]
print('   Q5-Q1 spread at 5d:  %+.2f%% ann.' % q5q1_5)
print('   Q5-Q1 spread at 21d: %+.2f%% ann.' % q5q1_21)
print('   Q5-Q1 spread at 63d: %+.2f%% ann.' % q5q1_63)
print()
print('4. Correlation with MOVE:')
print('   Corr(RR_1M, MOVE) = %.4f' % r3)
print('   Corr(z_std, MOVE)  = %.4f' % r2)
print('   Partial corr(RR_1M, TLT_fwd | MOVE) = %s' % pc_str)
print()
print('5. Signal Comparison:')
print('   z_std  Sharpe = %.3f  Excess = %.4f' % (m_std['sharpe'], m_std['excess']))
print('   z_move Sharpe = %.3f  Excess = %.4f' % (m_move['sharpe'], m_move['excess']))
print('   z_raw  Sharpe = %.3f  Excess = %.4f' % (m_raw['sharpe'], m_raw['excess']))
print('   z_rank Sharpe = %.3f  Excess = %.4f' % (m_rank['sharpe'], m_rank['excess']))
print()
print('=' * 72)
print('DIAGNOSTIC ANALYSIS COMPLETE')
print('=' * 72)
