
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Axial Length Percentile Calculator", layout="centered")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    expected = {"sex", "age", "axial length"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df["sex"] = df["sex"].astype(str).str.upper().str.strip().str[0]
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["axial length"] = pd.to_numeric(df["axial length"], errors="coerce")
    df = df.dropna(subset=["sex", "age", "axial length"])
    if np.all(np.equal(np.mod(df["age"], 1), 0)):
        df["age"] = df["age"].astype(int)
    return df

def ecdf_percentile(values: np.ndarray, x: float, method: str = "weak") -> float:
    n = len(values)
    if n == 0:
        return np.nan
    if method == "strict":
        rank = np.sum(values < x)
    else:
        rank = np.sum(values <= x)
    return 100.0 * rank / n

def parametric_percentile(values: np.ndarray, x: float) -> float:
    mu = np.mean(values)
    sd = np.std(values, ddof=1)
    if sd <= 0 or not np.isfinite(sd):
        return np.nan
    z = (x - mu) / sd
    from math import erf, sqrt
    cdf = 0.5 * (1.0 + erf(z / sqrt(2)))
    return 100.0 * cdf

def summarize(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)) if values.size else np.nan,
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
        "p5": float(np.percentile(values, 5)) if values.size else np.nan,
        "p10": float(np.percentile(values, 10)) if values.size else np.nan,
        "p25": float(np.percentile(values, 25)) if values.size else np.nan,
        "p50": float(np.percentile(values, 50)) if values.size else np.nan,
        "p75": float(np.percentile(values, 75)) if values.size else np.nan,
        "p90": float(np.percentile(values, 90)) if values.size else np.nan,
        "p95": float(np.percentile(values, 95)) if values.size else np.nan,
    }

st.title("Axial Length Percentile Calculator")
st.caption("나이·성별 집단에서 입력한 Axial Length의 백분위수를 계산하고, 연령별 퍼센타일 곡선을 시각화합니다.")

with st.sidebar:
    st.header("데이터")
    data_path = st.text_input("CSV 경로", value="data.csv", help="열 이름: sex, age, axial length")
    st.markdown("—")
    st.header("옵션")
    include_neighbor = st.checkbox("근처 연령 포함 (±1세)", value=False)
    tie_method = st.radio("동값 처리 방식", options=["같거나 작음(≤)", "작음(<)"], index=0, horizontal=True)
    method = st.radio("퍼센타일 방법", options=["경험적(ECDF)", "정규 가정(Parametric)"], index=0)

df = None
err = None
try:
    df = load_data(data_path)
except Exception as e:
    err = str(e)

if err:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {err}")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        unique_ages = sorted(df["age"].dropna().unique().tolist())
        default_age = unique_ages[len(unique_ages)//2] if unique_ages else 10
        age = st.number_input("나이 (세)", min_value=int(min(unique_ages)) if unique_ages else 0,
                              max_value=int(max(unique_ages)) if unique_ages else 100, value=int(default_age), step=1)
    with col2:
        sex = st.selectbox("성별", options=["M", "F"])
    with col3:
        axial_length = st.number_input("Axial Length (mm)", min_value=10.0, max_value=40.0, step=0.01, format="%.2f")
        axial_length = round(float(axial_length), 2)

    if include_neighbor:
        age_mask = (df["age"] >= age - 1) & (df["age"] <= age + 1)
    else:
        age_mask = (df["age"] == age)
    cohort = df[age_mask & (df["sex"] == sex)]
    values = cohort["axial length"].to_numpy()

    st.markdown("### 결과")
    if values.size == 0:
        st.warning("해당 조건에 해당하는 표본이 없습니다. 옵션을 조정해 보세요.")
    else:
        if method.startswith("경험적"):
            percentile = ecdf_percentile(values, axial_length, method=("strict" if tie_method.startswith("작음") else "weak"))
        else:
            percentile = parametric_percentile(values, axial_length)

        stats = summarize(values)

        st.metric(label=f"{age}세 {sex} 백분위수", value=f"{percentile:.2f} %")
        st.write(f"표본수 **n={stats['n']}**, 평균 **{stats['mean']:.2f} mm**, 표준편차 **{stats['sd']:.2f} mm**")

        bands = pd.DataFrame({
            "Percentile": ["P5","P10","P25","P50","P75","P90","P95"],
            "Axial Length (mm)": [
                round(stats["p5"], 2), round(stats["p10"], 2), round(stats["p25"], 2),
                round(stats["p50"], 2), round(stats["p75"], 2), round(stats["p90"], 2), round(stats["p95"], 2)
            ],
        })
        st.dataframe(bands, use_container_width=True)

        fig, ax = plt.subplots()
        ax.hist(values, bins="auto")
        ax.axvline(axial_length, color="red", linestyle="--")
        ax.set_xlabel("Axial Length (mm)")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution: Age {age}, Sex {sex} (n={stats['n']})")
        st.pyplot(fig)

        sex_df = df[df["sex"] == sex].copy()

        def pct_series(x):
            arr = x.to_numpy()
            if arr.size < 5:
                return pd.Series({"p5": np.nan, "p10": np.nan, "p25": np.nan, "p50": np.nan,
                                  "p75": np.nan, "p90": np.nan, "p95": np.nan})
            return pd.Series({
                "p5": np.percentile(arr, 5),
                "p10": np.percentile(arr, 10),
                "p25": np.percentile(arr, 25),
                "p50": np.percentile(arr, 50),
                "p75": np.percentile(arr, 75),
                "p90": np.percentile(arr, 90),
                "p95": np.percentile(arr, 95),
            })

        ref = sex_df.groupby("age")["axial length"].apply(pct_series).reset_index()
        ref = ref.pivot(index="age", columns="level_1", values="axial length").sort_index()

        fig2, ax2 = plt.subplots()
        for col in ["p5","p10","p25","p50","p75","p90","p95"]:
            if col in ref.columns and ref[col].notna().sum() > 0:
                ax2.plot(ref.index.values, ref[col].values, label=col.upper())
        ax2.scatter([age], [axial_length], s=60, zorder=5, color="red")
        ax2.set_xlabel("Age (years)")
        ax2.set_ylabel("Axial Length (mm)")
        ax2.set_title(f"Percentile Curves by Age — Sex {sex}")
        ax2.legend(loc="best")
        st.pyplot(fig2)

        with st.expander("표본 데이터 보기"):
            st.dataframe(cohort[["sex","age","axial length"]].sort_values("axial length").reset_index(drop=True), use_container_width=True)

st.markdown("---")
st.markdown("**사용 방법**")
st.markdown("1) 왼쪽 사이드바에서 CSV 경로와 옵션을 설정합니다. 2) 나이·성별·Axial Length를 입력합니다. 3) 결과 카드, 분포, 성장차트식 곡선을 확인합니다.")
