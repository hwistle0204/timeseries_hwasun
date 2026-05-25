# =====================================================================
# 전남 화순 딸기 총출하량 일별 시계열 분석 (ARMA)
# - 기존 노트북 흐름 유지 + asfreq('D') 등간격화 / 육묘기간 결측 처리 추가
# - ARIMA는 신버전 API(statsmodels.tsa.arima.model) 사용
# =====================================================================
import sys

print(sys.executable)
print(sys.version)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
import statsmodels.api as sm

plt.rc('font', family='NanumBarunGothic')   # 코랩 한글 폰트
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────────────────────────────
# 1. 데이터 로드 및 결측 처리
#    원본 CSV 인코딩 명시(cp949/utf-8 환경에 맞게 지정), 숫자 컬럼만 보간
# ─────────────────────────────────────────────────────────────────────
chonnam = pd.read_csv("전라남도22.csv", encoding='utf-8')

print(chonnam.isnull().sum())                       # 결측 분포 사전 점검
num_cols = chonnam.select_dtypes(include='number').columns
chonnam[num_cols] = chonnam[num_cols].interpolate() # 숫자 컬럼 선형 보간


# ─────────────────────────────────────────────────────────────────────
# 2. datetime 변환 및 화순 지역 필터링
# ─────────────────────────────────────────────────────────────────────
chonnam['datetime'] = pd.to_datetime(chonnam['출하일자'])

Hwasun = chonnam[chonnam['시군'] == '화순'].set_index('datetime')
Hwasun.drop(['도', '시군'], axis=1, inplace=True)


# ─────────────────────────────────────────────────────────────────────
# 3. 동일 일자 복수 농가 관측치를 일별 단일 값으로 집계 (최댓값)
# ─────────────────────────────────────────────────────────────────────
Hwasun1 = Hwasun.groupby(level=0).agg('max')[['총출하량']]
print("일별 집계 후 길이:", len(Hwasun1))            # 587


# ─────────────────────────────────────────────────────────────────────
# 4. [추가] asfreq('D') 등간격화 + 육묘기간 결측 처리
#    - 불규칙 인덱스를 일(D) 등간격으로 변환해 ARMA 등간격 가정 충족
#    - 출하가 구조적으로 없는 육묘기간(6~10월)은 분석 구간에서 제외
#    - 출하 시즌(11~5월) 내부의 단기 결측만 선형 보간
# ─────────────────────────────────────────────────────────────────────
Hwasun_daily = Hwasun1.asfreq('D')                   # 587일 -> 1289일(빈 날짜 NaN)

# 월별 결측률 확인 → 육묘기간 식별 (6~10월이 98~100%)
na_rate = Hwasun_daily.groupby(Hwasun_daily.index.month)['총출하량'] \
                      .apply(lambda s: s.isna().mean())
print("\n월별 결측률:\n", na_rate.round(2))

# 육묘기간(6~10월) 제외 → 출하 시즌만
Hwasun_season = Hwasun_daily[~Hwasun_daily.index.month.isin([6, 7, 8, 9, 10])].copy()

# 시즌 내부 단기 결측만 선형 보간 (ffill 아닌 linear: 출하량 추세 반영)
Hwasun_season['총출하량'] = Hwasun_season['총출하량'].interpolate(method='linear')

# 양끝 잔여 결측 제거 → 최종 등간격 시계열
Hwasun_df = Hwasun_season.dropna()
print("\n최종 분석 일수:", len(Hwasun_df))            # 830

# ARMA 입력용 시계열 (시즌 제외로 인덱스가 불연속이므로 정수 인덱스로 재설정)
ts = Hwasun_df['총출하량'].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# 5. 원시계열 시각화
# ─────────────────────────────────────────────────────────────────────
sns.set(rc={'figure.figsize': (21, 7)})
sns.lineplot(x=Hwasun_df.index, y=Hwasun_df['총출하량'])
plt.title("화순 딸기 총출하량 (등간격화 후)")
plt.show()


# ─────────────────────────────────────────────────────────────────────
# 6. 로그 변환 (분산 비정상성 완화)
# ─────────────────────────────────────────────────────────────────────
log_ts = np.log(ts)

sns.lineplot(x=log_ts.index, y=log_ts.values)
plt.title("로그 변환 시계열")
plt.show()


# ─────────────────────────────────────────────────────────────────────
# 7. 정상성 검정 (ADF / KPSS 교차 검정)
# ─────────────────────────────────────────────────────────────────────
print("\n[ADF] 원시계열  p =", round(adfuller(ts)[1], 4))
print("[ADF] 로그변환  p =", round(adfuller(log_ts)[1], 4))
print("[KPSS] 로그변환 p =", round(kpss(log_ts, nlags='auto')[1], 4))
# ADF: 귀무가설=비정상 / KPSS: 귀무가설=정상 → 두 검정 교차 확인


# ─────────────────────────────────────────────────────────────────────
# 8. ACF / PACF 시각화
# ─────────────────────────────────────────────────────────────────────
plot_acf(log_ts)
plot_pacf(log_ts)
plt.show()


# ─────────────────────────────────────────────────────────────────────
# 9. EACF (ARMA 후보 차수 탐색)
# ─────────────────────────────────────────────────────────────────────
def esacf(data, ar_max=7, ma_max=13, alpha=0.05, symbol=True):
    sig = norm.ppf(1 - alpha / 2)

    def lag_function(data, lag=1):
        res = [np.nan] * lag + list(data[:-lag])
        return np.array(res)

    def ar_ols(data, ar_order):
        depedent_data = np.array(data[ar_order:])
        X = np.empty((0, ar_order))
        for i in range(ar_order, len(data)):
            temp_row = data[i - ar_order:i][::-1]
            X = np.vstack([X, temp_row])
        results = sm.OLS(depedent_data, X).fit()
        return results.params

    def reupm(mat, ncol):
        k = ncol - 1
        for i in range(k):
            i1 = i + 1
            work = lag_function(mat[:, i])
            work[0] = -1
            temp = mat[:, i1] - (mat[i1, i1] / mat[i, i]) * work
            temp[i1] = 0
            if i == 0:
                mat2 = np.expand_dims(temp, axis=1)
            else:
                mat2 = np.column_stack((mat2, temp))
        return mat2

    ar_max += 1
    ma_max += 1
    nar = ar_max - 1
    nma = ma_max
    ncov = nar + nma + 2
    nrow = nar + nma + 1
    ncol = nrow - 1

    def ceascf(m, cov1, nar, ncol, count, ncov, z, zm):
        result = [0] * (nar + 1)
        result[0] = cov1[ncov + count]
        for i in range(nar):
            temp = np.column_stack((z[i + 1:], zm[i + 1:, :i + 1])).dot([1] + list(-mat2[:i + 1, i]))
            result[i + 1] = acf(temp, nlags=count + 1, fft=False)[count + 1]
        return result

    z = data - np.mean(data)
    for i in range(nar):
        if i == 0:
            zm = np.expand_dims(lag_function(z, i + 1), axis=1)
        else:
            zm = np.column_stack((zm, lag_function(z, i + 1)))

    cov1 = acf(z, nlags=ncov, fft=False)
    cov1 = np.array(list(cov1[1:][::-1]) + list(cov1))
    ncov += 1
    mat = np.zeros((nrow, ncol))
    for i in range(ncol):
        mat[:i + 1, i] = ar_ols(z, ar_order=i + 1)

    for i in range(nma):
        mat2 = reupm(mat, ncol)
        ncol = ncol - 1
        if i == 0:
            eacfm = np.expand_dims(ceascf(mat2, cov1, nar, ncol, i, ncov, z, zm), axis=1)
        else:
            eacfm = np.column_stack((eacfm, ceascf(mat2, cov1, nar, ncol, i, ncov, z, zm)))
        mat = mat2

    if symbol:
        work = len(z) - np.array(range(1, nar + 2)) + 1
        for j in range(nma):
            work = work - 1
            temp = np.abs(eacfm[:, j]) > sig / np.sqrt(work)
            temp = np.array(['X' if t else 'O' for t in temp])
            if j == 0:
                sym = np.expand_dims(temp, axis=1)
            else:
                sym = np.column_stack((sym, temp))
        return pd.DataFrame(sym)
    return pd.DataFrame(eacfm)

print("\n[EACF]")
print(esacf(log_ts.values, ar_max=7, ma_max=7, alpha=0.05, symbol=True))


# ─────────────────────────────────────────────────────────────────────
# 10. AIC 기반 차수 Grid Search (p, q = 0~2)
# ─────────────────────────────────────────────────────────────────────
aic_val, order_list = [], []
for ar in range(3):
    for ma in range(3):
        order = (ar, 0, ma)
        order_list.append(str(order))
        aic_val.append(ARIMA(log_ts, order=order).fit().aic)

fig = plt.figure(figsize=(10, 6))
fig.set_facecolor('white')
plt.plot(range(len(aic_val)), aic_val)
plt.xticks(range(len(aic_val)), order_list, rotation=90)
plt.title("차수 조합별 AIC")
plt.show()

order_aic = dict(zip(order_list, aic_val))
final_order, final_aic = sorted(order_aic.items(), key=lambda x: x[1])[0]
print(f"\n최소 AIC 차수: {final_order}, AIC: {final_aic:.1f}")
# 최소 AIC는 ARMA(1,2)이나, ARMA(1,1)과 차이가 작아
# 모수 절약(parsimony) 원칙에 따라 더 단순한 ARMA(1,1)을 최종 선택


# ─────────────────────────────────────────────────────────────────────
# 11. 최종 모형 ARMA(1,1) 적합
# ─────────────────────────────────────────────────────────────────────
final_model = ARIMA(log_ts, order=(1, 0, 1)).fit()
print(final_model.summary())


# ─────────────────────────────────────────────────────────────────────
# 12. 7시차 예측 (예측값 / 표준편차 / 95% 신뢰구간)
# ─────────────────────────────────────────────────────────────────────
fc = final_model.get_forecast(steps=7)
pred_mean = fc.predicted_mean
pred_ci = fc.conf_int(alpha=0.05)

print("\n7시차 예측(로그):\n", pred_mean.round(3))
print("7시차 예측(원단위):\n", np.exp(pred_mean).round(1))


# ─────────────────────────────────────────────────────────────────────
# 13. 예측 시각화 (관측 + 적합 + 예측구간)
# ─────────────────────────────────────────────────────────────────────
fitted_values = final_model.fittedvalues
n = len(log_ts)
pred_x = range(n, n + 7)

plt.figure(figsize=(10, 6))
plt.plot(range(n), log_ts.values, label='data')
plt.plot(range(n), fitted_values, label='fitted')
plt.plot(pred_x, pred_mean.values, color='red', label='prediction')
plt.plot(pred_x, pred_ci.iloc[:, 0], color='red', linestyle='--', linewidth=1)
plt.plot(pred_x, pred_ci.iloc[:, 1], color='red', linestyle='--', linewidth=1)
plt.fill_between(pred_x, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='red', alpha=0.2)
plt.legend()
plt.title("화순 딸기 총출하량 ARMA(1,1) 7시차 예측")
plt.show()
