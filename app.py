import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import time, timedelta
from scipy import stats as scipy_stats
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Travel Time Predictor", page_icon="🚦", layout="wide")

# --- Helper Functions ---

def build_input(start_sec, bottleneck, vehicle_class, avg_vehicles, training_columns):
    """Build a model-ready input DataFrame from user parameters."""
    input_dict = {
        'Start Time (Elapsed Sec)': start_sec,
        'Start_Time_Elapsed_Sec': start_sec,
        'Bottleneck Delay (sec)': bottleneck,
        'Bottleneck_Delay_Sec': bottleneck,
        'Total Vehicles': avg_vehicles
    }
    for col in training_columns:
        if col.startswith('Class_'):
            class_name = col.replace('Class_', '')
            input_dict[col] = 1 if class_name.lower() == vehicle_class.lower() else 0
    final = {k: v for k, v in input_dict.items() if k in training_columns}
    df = pd.DataFrame([final], columns=training_columns).fillna(0)
    return df

def predict_with_floor(model, input_df, bottleneck):
    """Run prediction and apply physics floor."""
    preds = model.predict(input_df)[0]
    p500, p1km = preds[0], preds[1]
    if p1km < p500 + bottleneck + 15:
        p1km = p500 + bottleneck + 15
    return p500, p1km

def get_congestion_label(pred_1km_sec):
    """Classify congestion severity based on 1km travel time."""
    if pred_1km_sec < 150:
        return "🟢 Low", "green"
    elif pred_1km_sec < 250:
        return "🟡 Moderate", "orange"
    elif pred_1km_sec < 350:
        return "🟠 Heavy", "red"
    else:
        return "🔴 Gridlock", "darkred"

def get_delay_breakdown(model, input_df, training_columns):
    """Decompose prediction into per-feature contributions."""
    coeffs = model.coef_[1]  # 1km coefficients
    values = input_df.values[0]
    breakdown = {}
    for i, col in enumerate(training_columns):
        contribution = coeffs[i] * values[i]
        if abs(contribution) > 0.5:
            breakdown[col] = contribution
    return breakdown

def format_clock(elapsed_sec):
    """Convert elapsed peak-hour seconds to a clock string."""
    h = 17 + elapsed_sec // 3600
    m = (elapsed_sec % 3600) // 60
    period = "PM" if h < 24 else "AM"
    display_h = h if h <= 12 else h - 12
    return f"{display_h}:{m:02d} {period}"

# --- Load Backend ---

@st.cache_resource
def load_backend():
    model = joblib.load('traffic_model.pkl')
    columns = joblib.load('model_columns.pkl')
    df = pd.read_csv('Master_Traffic_Data.csv')
    hv_col = 'HVs_Encountered' if 'HVs_Encountered' in df.columns else 'HVs Encountered'
    lv_col = 'LVs_Encountered' if 'LVs_Encountered' in df.columns else 'LVs Encountered'
    avg_total_vehicles = df[hv_col].mean() + df[lv_col].mean()

    # Replicate train_model.py feature engineering for stats
    df_clean = df[~df['Remarks'].str.contains('Raw Data', na=False)].copy()
    numeric_cols = ['Start Time (Elapsed Sec)', 'Time to 500m (sec)',
                    'Bottleneck Delay (sec)', hv_col, lv_col, 'Time to 1km (sec)']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean['Total Vehicles'] = df_clean[hv_col] + df_clean[lv_col]
    df_clean = df_clean.drop(columns=[hv_col, lv_col])
    df_model = df_clean.drop(columns=['Vehicle No.', 'Actual Start Time', 'Remarks'])
    df_model = pd.get_dummies(df_model, columns=['Class'], drop_first=False)
    df_model = df_model[df_model['Time to 500m (sec)'] < 400].dropna()

    X = df_model.drop(columns=['Time to 500m (sec)', 'Time to 1km (sec)'])
    y = df_model[['Time to 500m (sec)', 'Time to 1km (sec)']]

    return model, columns, avg_total_vehicles, X, y

try:
    model, training_columns, avg_total_vehicles, X_data, y_data = load_backend()
except FileNotFoundError:
    st.error("⚠️ Backend files missing! Please run train_model.py first.")
    st.stop()

# --- UI ---

st.title("🚦 Travel Time Predictor")
st.markdown("##### Multi-Milestone AI Prediction & Congestion Analysis")
st.divider()

col_input, col_result = st.columns([1.2, 1])

with col_input:
    st.subheader("📋 Input Travel Conditions")
    with st.container(border=True):
        vehicle_class = st.selectbox("Select Vehicle Type", ['Car', 'Truck', 'Van', 'Bus', 'Bike', 'Auto', 'Scooter'])

        departure_time = st.slider(
            "Departure Time (Evening Peak)",
            min_value=time(17, 0),
            max_value=time(19, 0),
            value=time(17, 30),
            format="hh:mm a"
        )
        start_time_sec = (departure_time.hour - 17) * 3600 + departure_time.minute * 60

        bottleneck_delay = st.slider("Active Bottleneck Delay (Seconds)", 0, 60, 5)

    st.info(f"📊 **Data Insight:** Model using dataset average of **{int(avg_total_vehicles)} total vehicles** for density.")

with col_result:
    st.subheader("⏱️ Predicted ETAs")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Forecast Commute Timeline", type="primary", use_container_width=True):

        input_df = build_input(start_time_sec, bottleneck_delay, vehicle_class, avg_total_vehicles, training_columns)
        pred_500m, pred_1km = predict_with_floor(model, input_df, bottleneck_delay)

        # --- Dual Metric Cards ---
        m1, m2 = st.columns(2)
        with m1:
            with st.container(border=True):
                st.metric("ETA to 500m", f"{int(pred_500m)} Sec", delta=f"{pred_500m/60:.1f} Min", delta_color="off")
        with m2:
            with st.container(border=True):
                st.metric("ETA to 1km", f"{int(pred_1km)} Sec", delta=f"{pred_1km/60:.1f} Min", delta_color="off")

        # --- Congestion Severity ---
        label, color = get_congestion_label(pred_1km)
        st.markdown(f"**Congestion Level:** <span style='color:{color}; font-size:1.3em;'>{label}</span>", unsafe_allow_html=True)

        st.success("✅ Multi-output prediction generated successfully.")

        # --- Delay Breakdown ---
        st.divider()
        st.subheader("📊 Delay Breakdown")

        breakdown = get_delay_breakdown(model, input_df, training_columns)
        # Group into readable categories
        base_time = sum(v for k, v in breakdown.items() if k.startswith('Class_'))
        time_factor = breakdown.get('Start Time (Elapsed Sec)', breakdown.get('Start_Time_Elapsed_Sec', 0))
        bn_factor = breakdown.get('Bottleneck Delay (sec)', breakdown.get('Bottleneck_Delay_Sec', 0))
        density_factor = breakdown.get('Total Vehicles', 0)

        bd_df = pd.DataFrame({
            'Factor': ['Vehicle Base Time', 'Departure Timing', 'Bottleneck Impact', 'Traffic Density'],
            'Contribution (sec)': [round(base_time, 1), round(time_factor, 1), round(bn_factor, 1), round(density_factor, 1)]
        })
        st.bar_chart(bd_df.set_index('Factor'), horizontal=True)
        st.caption("How each factor contributes to your predicted 1km travel time.")

        # --- Best Departure Suggestion ---
        st.divider()
        st.subheader("🕐 Best Departure Time")

        times_sec = list(range(0, 7201, 300))  # every 5 min across 5-7 PM
        results = []
        for t in times_sec:
            inp = build_input(t, bottleneck_delay, vehicle_class, avg_total_vehicles, training_columns)
            _, p1km = predict_with_floor(model, inp, bottleneck_delay)
            results.append({'Elapsed': t, 'Predicted 1km Time (sec)': round(p1km, 1)})

        sweep_df = pd.DataFrame(results)
        best_idx = sweep_df['Predicted 1km Time (sec)'].idxmin()
        best_elapsed = sweep_df.loc[best_idx, 'Elapsed']
        best_time = sweep_df.loc[best_idx, 'Predicted 1km Time (sec)']

        st.markdown(f"For a **{vehicle_class}** with **{bottleneck_delay}s** bottleneck delay, the optimal departure is **{format_clock(int(best_elapsed))}** "
                    f"with an estimated 1km time of **{int(best_time)} sec ({best_time/60:.1f} min)**.")

        # --- Peak Hour Chart ---
        chart_df = sweep_df.copy()
        chart_df['Departure Time'] = chart_df['Elapsed'].apply(lambda x: format_clock(int(x)))
        chart_df = chart_df.set_index('Departure Time')

        st.line_chart(chart_df['Predicted 1km Time (sec)'], use_container_width=True)
        st.caption("Predicted 1km travel time across the 5:00–7:00 PM evening peak window.")

    else:
        m1, m2 = st.columns(2)
        with m1:
            with st.container(border=True):
                st.metric("ETA to 500m", "-- Sec", delta="-- Min", delta_color="off")
        with m2:
            with st.container(border=True):
                st.metric("ETA to 1km", "-- Sec", delta="-- Min", delta_color="off")
        st.caption("Select your departure time and click forecast.")

# =============================================================
# STATISTICAL ANALYSIS SECTION
# =============================================================
st.divider()
st.header("📐 Statistical Analysis")

# --- 1. Model Health Panel ---
with st.expander("🏥 Model Health — R², Adjusted R², RMSE, F-Statistic", expanded=True):

    y_pred_all = model.predict(X_data.values.astype(float))
    n = len(X_data)
    p = X_data.shape[1]  # number of features

    # Per-target metrics
    r2_500 = r2_score(y_data.iloc[:, 0].astype(float), y_pred_all[:, 0])
    r2_1km = r2_score(y_data.iloc[:, 1].astype(float), y_pred_all[:, 1])
    r2_overall = r2_score(y_data.astype(float), y_pred_all)

    # Adjusted R² (for 1km target)
    adj_r2 = 1 - (1 - r2_1km) * (n - 1) / (n - p - 1)

    # RMSE
    rmse_500 = np.sqrt(mean_squared_error(y_data.iloc[:, 0].astype(float), y_pred_all[:, 0]))
    rmse_1km = np.sqrt(mean_squared_error(y_data.iloc[:, 1].astype(float), y_pred_all[:, 1]))

    # F-Statistic for 1km model: F = (R² / p) / ((1 - R²) / (n - p - 1))
    f_stat = (r2_1km / p) / ((1 - r2_1km) / (n - p - 1)) if r2_1km < 1 else float('inf')
    f_pvalue = 1 - scipy_stats.f.cdf(f_stat, p, n - p - 1)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("R² (Overall)", f"{r2_overall:.4f}")
    mc2.metric("Adjusted R² (1km)", f"{adj_r2:.4f}")
    mc3.metric("RMSE — 500m", f"{rmse_500:.2f} sec")
    mc4.metric("RMSE — 1km", f"{rmse_1km:.2f} sec")

    mc5, mc6, mc7, mc8 = st.columns(4)
    mc5.metric("R² — 500m", f"{r2_500:.4f}")
    mc6.metric("R² — 1km", f"{r2_1km:.4f}")
    mc7.metric("F-Statistic", f"{f_stat:.2f}")
    mc8.metric("F-test p-value", f"{f_pvalue:.2e}")

    st.caption("R² closer to 1 = better fit. F-statistic tests whether the model is statistically significant overall. p < 0.05 means the model is significant.")

# --- 2. Hypothesis Testing / Coefficient Significance ---
with st.expander("🔬 Hypothesis Testing — Coefficient Significance (t-tests)"):

    st.markdown("**H₀:** Coefficient = 0 (feature has no effect) &nbsp;|&nbsp; **H₁:** Coefficient ≠ 0 (feature matters)")
    st.markdown("Reject H₀ if **p-value < 0.05** (95% confidence level)")

    # Compute t-statistics and p-values for 1km target coefficients
    residuals = y_data.iloc[:, 1].values.astype(float) - y_pred_all[:, 1]
    mse_resid = np.sum(residuals ** 2) / (n - p)
    X_arr = X_data.values.astype(float)

    # Variance-covariance matrix of coefficients
    try:
        XtX_inv = np.linalg.inv(X_arr.T @ X_arr)
        se_coeffs = np.sqrt(np.diag(XtX_inv) * mse_resid)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X_arr.T @ X_arr)
        se_coeffs = np.sqrt(np.abs(np.diag(XtX_inv)) * mse_resid)

    coeffs_1km = model.coef_[1]
    t_stats = coeffs_1km / (se_coeffs + 1e-10)
    p_values = 2 * (1 - scipy_stats.t.cdf(np.abs(t_stats), df=n - p))

    sig_df = pd.DataFrame({
        'Feature': list(training_columns),
        'Coefficient': np.round(coeffs_1km, 4),
        'Std Error': np.round(se_coeffs, 4),
        't-Statistic': np.round(t_stats, 3),
        'p-value': np.round(p_values, 5),
        'Significant (p<0.05)': ['✅ Yes' if pv < 0.05 else '❌ No' for pv in p_values]
    })

    st.dataframe(sig_df, use_container_width=True, hide_index=True)
    sig_count = sum(1 for pv in p_values if pv < 0.05)
    st.caption(f"{sig_count} out of {p} features are statistically significant at the 95% confidence level.")

# --- 3. Confidence Intervals ---
with st.expander("📏 Confidence Intervals — 95% Prediction Intervals"):

    st.markdown("Shows the 95% confidence interval for the **mean predicted value** across the dataset.")

    y_pred_1km = y_pred_all[:, 1]
    pred_mean = np.mean(y_pred_1km)
    pred_se = np.std(y_pred_1km, ddof=1) / np.sqrt(n)
    t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
    ci_lower = pred_mean - t_crit * pred_se
    ci_upper = pred_mean + t_crit * pred_se

    ci1, ci2, ci3 = st.columns(3)
    ci1.metric("Mean Predicted 1km Time", f"{pred_mean:.1f} sec")
    ci2.metric("95% CI Lower Bound", f"{ci_lower:.1f} sec")
    ci3.metric("95% CI Upper Bound", f"{ci_upper:.1f} sec")

    # Also show prediction interval (wider, accounts for individual prediction uncertainty)
    pred_std = np.sqrt(mse_resid + pred_se**2)
    pi_lower = pred_mean - t_crit * pred_std
    pi_upper = pred_mean + t_crit * pred_std

    pi1, pi2, pi3 = st.columns(3)
    pi1.metric("Residual Std Error", f"{np.sqrt(mse_resid):.1f} sec")
    pi2.metric("95% PI Lower Bound", f"{pi_lower:.1f} sec")
    pi3.metric("95% PI Upper Bound", f"{pi_upper:.1f} sec")

    st.caption("CI = uncertainty in the mean prediction. PI = uncertainty for an individual new prediction (wider).")

# --- 4. Sampling Summary ---
with st.expander("📊 Sampling Summary — Dataset Statistics"):

    st.markdown(f"**Total Observations (n):** {n}")
    st.markdown(f"**Number of Features (p):** {p}")
    st.markdown(f"**Degrees of Freedom:** {n - p}")
    clt_met = "✅ Yes (n ≥ 30)" if n >= 30 else "❌ No (n < 30)"
    st.markdown(f"**Central Limit Theorem (n ≥ 30):** {clt_met}")

    st.markdown("---")
    st.markdown("**Target Variable Summary (Time to 1km):**")

    target = y_data.iloc[:, 1].astype(float)
    ss1, ss2, ss3, ss4 = st.columns(4)
    ss1.metric("Mean", f"{target.mean():.1f} sec")
    ss2.metric("Std Deviation", f"{target.std():.1f} sec")
    ss3.metric("Standard Error", f"{target.std() / np.sqrt(n):.2f} sec")
    ss4.metric("Median", f"{target.median():.1f} sec")

    ss5, ss6, ss7, ss8 = st.columns(4)
    ss5.metric("Min", f"{target.min():.0f} sec")
    ss6.metric("Max", f"{target.max():.0f} sec")
    ss7.metric("Skewness", f"{target.skew():.3f}")
    ss8.metric("Kurtosis", f"{target.kurtosis():.3f}")

    st.caption("Standard Error = Std Dev / √n. Skewness near 0 = symmetric distribution. Kurtosis near 0 = normal-like tails.")

# --- 5. Correlation Matrix ---
with st.expander("🔗 Correlation Matrix — Feature Relationships"):

    # Combine X and y for full correlation
    corr_df = pd.concat([X_data, y_data], axis=1).apply(pd.to_numeric, errors='coerce')
    corr_matrix = corr_df.corr()

    # Display as styled heatmap
    st.markdown("**Pearson Correlation Coefficients** across all features and targets:")
    styled = corr_matrix.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1).format("{:.2f}")
    st.dataframe(styled, use_container_width=True)

    # Highlight key insight
    if 'Total Vehicles' in corr_matrix.columns and 'Time to 1km (sec)' in corr_matrix.columns:
        tv_corr = corr_matrix.loc['Total Vehicles', 'Time to 1km (sec)']
        st.markdown(f"**Key Insight:** Correlation between Total Vehicles and 1km Time = **{tv_corr:.3f}** — "
                    f"{'moderate positive' if tv_corr > 0.3 else 'weak'} relationship, justifying its inclusion as a feature.")

    st.caption("Values near +1/-1 indicate strong positive/negative linear relationships. The HVs + LVs merge into Total Vehicles was done to eliminate the multicollinearity visible in the original matrix.")