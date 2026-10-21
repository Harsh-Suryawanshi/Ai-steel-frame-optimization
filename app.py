"""
SteelForm.AI: Enterprise Machine Learning for Preliminary Sizing of Multistory Steel Frames
Engineered to IS 800:2007 (Limit State Design) and IS 1893:2016 (Seismic Criteria).
"""

import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from etabs_staad_bridge import generate_etabs_e2k, generate_staad_std, get_etabs_oapi_batch_template
from predictor import MemberSizingPredictor
from steel_sections import BEAM_SECTIONS, COLUMN_SECTIONS
from visualizer import (
    create_2d_elevation_fig,
    create_3d_frame_fig,
    create_section_geometry_fig,
    create_seismic_distribution_fig,
    create_story_steel_takeoff_fig,
    create_feature_importance_fig,
)

# Page configuration
st.set_page_config(
    page_title="SteelForm.AI | Multistory Frame Sizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enterprise CSS Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Top Header Container */
    .brand-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 24px;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .brand-title {
        color: #F8FAFC;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .brand-badge {
        background: rgba(14, 165, 233, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .brand-sub {
        color: #94A3B8;
        font-size: 0.88rem;
        margin-top: 4px;
    }

    /* Cards & Containers */
    .pro-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        margin-bottom: 16px;
    }
    .pro-card-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 10px;
    }

    /* Status Badges */
    .badge-pass {
        background-color: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-warn {
        background-color: #FFFBEB;
        color: #D97706;
        border: 1px solid #FDE68A;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
    }

    /* KPI Cards */
    .kpi-container {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 16px;
        text-align: left;
    }
    .kpi-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        margin: 4px 0;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #64748B;
    }

    /* Code block styling */
    pre, code {
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return MemberSizingPredictor()


predictor = load_predictor()

# --- BRAND BANNER HEADER ---
st.markdown("""
<div class="brand-container">
    <div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <span class="brand-title">STEEL<span style="color: #38BDF8;">FORM</span>.AI</span>
            <span class="brand-badge">IS 800:2007 &amp; IS 1893</span>
        </div>
        <div class="brand-sub">Physics-Informed Machine Learning for Rapid Preliminary Sizing of Multistory Steel Frames</div>
    </div>
    <div style="text-align: right; color: #94A3B8; font-size: 0.82rem;">
        <div><b>Standard:</b> Fe 410 Steel (E250)</div>
        <div><b>Model Engine:</b> MLP NeuralNet &amp; RF Regressor</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR: GLOBAL BUILDING CONFIGURATION ---
st.sidebar.markdown("### Building Configuration")

# Preset selector
preset = st.sidebar.selectbox(
    "Load Engineering Preset",
    ["Custom Configuration", "Standard G+10 Office (Zone IV)", "Critical Hospital G+10 (Zone V, I=1.5)", "Commercial Hub G+6 (Zone III)"],
    index=0,
)

if preset == "Standard G+10 Office (Zone IV)":
    p_stories, p_height, p_bx, p_by, p_dl, p_ll, p_zone, p_soil, p_r, p_i = 10, 3.5, 6.0, 6.0, 4.5, 3.0, "Zone IV", "Type II (Medium)", 5.0, 1.2
elif preset == "Critical Hospital G+10 (Zone V, I=1.5)":
    p_stories, p_height, p_bx, p_by, p_dl, p_ll, p_zone, p_soil, p_r, p_i = 10, 3.8, 7.0, 7.0, 5.0, 4.0, "Zone V", "Type II (Medium)", 5.0, 1.5
elif preset == "Commercial Hub G+6 (Zone III)":
    p_stories, p_height, p_bx, p_by, p_dl, p_ll, p_zone, p_soil, p_r, p_i = 6, 3.5, 6.0, 6.0, 4.0, 3.0, "Zone III", "Type I (Rock/Hard)", 5.0, 1.0
else:
    p_stories, p_height, p_bx, p_by, p_dl, p_ll, p_zone, p_soil, p_r, p_i = 10, 3.5, 6.0, 6.0, 4.5, 3.0, "Zone IV", "Type II (Medium)", 5.0, 1.2

n_stories = st.sidebar.slider("Number of Stories (G+N)", min_value=3, max_value=15, value=p_stories, step=1)
story_height = st.sidebar.slider("Story Height (m)", min_value=2.8, max_value=4.5, value=p_height, step=0.1)

col_sx, col_sy = st.sidebar.columns(2)
with col_sx:
    bay_x_count = st.number_input("Bays X", min_value=2, max_value=8, value=4, step=1)
    bay_x_length = st.number_input("Span X (m)", min_value=3.5, max_value=10.0, value=p_bx, step=0.5)
with col_sy:
    bay_y_count = st.number_input("Bays Y", min_value=2, max_value=8, value=3, step=1)
    bay_y_length = st.number_input("Span Y (m)", min_value=3.5, max_value=10.0, value=p_by, step=0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("### Loads & Seismic Actions")

dead_load = st.sidebar.slider("Floor Dead Load [kN/m²]", min_value=2.5, max_value=7.0, value=p_dl, step=0.25, help="Includes concrete deck slab, finishes, MEP & partitions")
live_load = st.sidebar.slider("Floor Live Load [kN/m²]", min_value=1.5, max_value=6.0, value=p_ll, step=0.5, help="Occupancy live load per IS 875 Part 2")

zone_options = ["Zone II (0.10g)", "Zone III (0.16g)", "Zone IV (0.24g)", "Zone V (0.36g)"]
zone_idx = 2 if p_zone == "Zone IV" else (3 if p_zone == "Zone V" else (1 if p_zone == "Zone III" else 0))
seismic_zone = st.sidebar.selectbox("IS 1893 Seismic Zone", zone_options, index=zone_idx)
zone_name = seismic_zone.split(" ")[0] + " " + seismic_zone.split(" ")[1]

with st.sidebar.expander("Advanced Seismic & Wind Options", expanded=False):
    soil_type = st.selectbox("Soil Profile Type", ["Type I (Rock/Hard)", "Type II (Medium)", "Type III (Soft)"], index=1 if "Type II" in p_soil else (0 if "Type I" in p_soil else 2))
    resp_r = st.selectbox("Response Reduction (R)", [4.0, 5.0], index=1 if p_r == 5.0 else 0, help="5.0 for Special Moment Frame (SMRF)")
    imp_i = st.selectbox("Importance Factor (I)", [1.0, 1.2, 1.5], index=1 if p_i == 1.2 else (2 if p_i == 1.5 else 0))
    wind_speed = st.slider("Basic Wind Speed Vb (m/s)", min_value=33.0, max_value=55.0, value=44.0, step=1.0)

# Global parameters dict
building_params = {
    "n_stories": int(n_stories),
    "story_height": float(story_height),
    "bay_x_count": int(bay_x_count),
    "bay_y_count": int(bay_y_count),
    "bay_x_length": float(bay_x_length),
    "bay_y_length": float(bay_y_length),
    "dead_load": float(dead_load),
    "live_load": float(live_load),
    "seismic_zone": zone_name,
    "soil_type": soil_type,
    "response_reduction_r": float(resp_r),
    "importance_factor_i": float(imp_i),
    "wind_speed": float(wind_speed),
}

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Member Sizing Studio",
    "Building Schedule & BIM Views",
    "Model Analytics & Diagnostics",
    "ETABS / STAAD.Pro Exporter",
])

# ==============================================================================
# TAB 1: MEMBER SIZING STUDIO
# ==============================================================================
with tab1:
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 1.2])
    with col_ctrl1:
        member_type = st.radio("Select Member Type", ["Beam", "Column"], horizontal=True)
    with col_ctrl2:
        target_story = st.selectbox("Story Level", [f"Story {s}" for s in range(1, n_stories + 1)], index=0)
        target_story_num = int(target_story.split(" ")[1])
    with col_ctrl3:
        if member_type == "Beam":
            member_loc = st.selectbox("Location / Role", ["Primary_Interior", "Primary_Exterior", "Secondary"], index=0)
        else:
            member_loc = st.selectbox("Location / Role", ["Interior", "Exterior", "Corner"], index=0)

    col_left, col_right = st.columns([1.25, 1.0])

    if member_type == "Beam":
        span_len = bay_x_length if "Primary" in member_loc else bay_y_length
        b_res = predictor.predict_beam({
            **building_params,
            "story_num": target_story_num,
            "beam_location": member_loc,
            "span": span_len,
        })

        with col_left:
            # Primary KPI Row
            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-title">Optimal Section</div>
                    <div class="kpi-value" style="color: #0284C7;">{b_res['recommended_section']}</div>
                    <div class="kpi-sub">{b_res['section_depth_mm']} mm Deep • {b_res['section_flange_width_mm']} mm Flange</div>
                </div>
                """, unsafe_allow_html=True)
            with k2:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-title">Plastic Modulus (Zp)</div>
                    <div class="kpi-value">{b_res['section_zp_cm3']:.0f} <span style="font-size: 0.9rem; color: #64748B;">cm³</span></div>
                    <div class="kpi-sub">Req: {b_res['predicted_zp_req_cm3']:.0f} cm³</div>
                </div>
                """, unsafe_allow_html=True)
            with k3:
                util = b_res["utilization_ratio"]
                badge_class = "badge-pass" if util <= 1.0 else "badge-warn"
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-title">Stress Ratio (Mu / Md)</div>
                    <div class="kpi-value">{util:.2f}</div>
                    <span class="{badge_class}">IS 800 {b_res['status']}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Detailed Engineering Breakdown Card
            st.markdown(f"""
            <div class="pro-card">
                <div class="pro-card-header">Limit State Design Verification (IS 800:2007 Cl. 8.2.1.2)</div>
                <table style="width: 100%; font-size: 0.88rem; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="color: #64748B; padding: 6px 0;">Beam Span (<i>L</i>)</td>
                        <td style="font-weight: 700; text-align: right;">{span_len:.2f} m</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="color: #64748B; padding: 6px 0;">Factored Bending Demand (<i>M</i><sub>u</sub>)</td>
                        <td style="font-weight: 700; text-align: right; color: #DC2626;">{b_res['factored_moment_demand_kNm']:.1f} kNm</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="color: #64748B; padding: 6px 0;">Design Bending Capacity (<i>M</i><sub>d</sub>)</td>
                        <td style="font-weight: 700; text-align: right; color: #059669;">{b_res['moment_capacity_kNm']:.1f} kNm</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="color: #64748B; padding: 6px 0;">Unit Steel Weight</td>
                        <td style="font-weight: 700; text-align: right;">{b_res['section_weight_kg_m']:.1f} kg/m</td>
                    </tr>
                    <tr>
                        <td style="color: #64748B; padding: 6px 0;">Deflection &amp; Stability Check</td>
                        <td style="font-weight: 700; text-align: right; color: #059669;">Passes <i>L</i>/325 criteria</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown("#### Scaled Engineering Cross-Section")
            sec_dict = next((s for s in BEAM_SECTIONS if s["name"] == b_res["recommended_section"]), BEAM_SECTIONS[-1])
            fig_sec = create_section_geometry_fig(sec_dict)
            st.pyplot(fig_sec)

    else:  # Column
        c_res = predictor.predict_column({
            **building_params,
            "story_num": target_story_num,
            "column_location": member_loc,
        })

        with col_left:
            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-title">Optimal Section</div>
                    <div class="kpi-value" style="color: #0284C7;">{c_res['recommended_section']}</div>
                    <div class="kpi-sub">{c_res['section_depth_mm']} mm Deep • {c_res['section_flange_width_mm']} mm Flange</div>
                </div>
                """, unsafe_allow_html=True)
            with k2:
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-title">Cross-Section Area</div>
                    <div class="kpi-value">{c_res['section_area_cm2']:.0f} <span style="font-size: 0.9rem; color: #64748B;">cm²</span></div>
                    <div class="kpi-sub">Req: {c_res['predicted_area_req_cm2']:.0f} cm²</div>
                </div>
                """, unsafe_allow_html=True)
            with k3:
                util = c_res["interaction_ratio"]
                badge_class = "badge-pass" if util <= 1.0 else "badge-warn"
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-title">Interaction Ratio</div>
                    <div class="kpi-value">{util:.2f}</div>
                    <span class="{badge_class}">IS 800 {c_res['status']}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="pro-card">
                <div class="pro-card-header">Axial &amp; Bending Verification (IS 800:2007 Cl. 7.1.2 &amp; Cl. 9.3)</div>
                <table style="width: 100%; font-size: 0.88rem; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="color: #64748B; padding: 6px 0;">Cumulative Axial Demand (<i>P</i><sub>u</sub>)</td>
                        <td style="font-weight: 700; text-align: right; color: #DC2626;">{c_res['factored_axial_demand_kN']:.1f} kN</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="color: #64748B; padding: 6px 0;">Design Compressive Capacity (<i>P</i><sub>d</sub>)</td>
                        <td style="font-weight: 700; text-align: right; color: #059669;">{c_res['axial_capacity_kN']:.1f} kN</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="color: #64748B; padding: 6px 0;">Effective Length (<i>L</i><sub>eff</sub>)</td>
                        <td style="font-weight: 700; text-align: right;">{story_height * 0.95:.2f} m (Buckling Class c)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="color: #64748B; padding: 6px 0;">Unit Steel Weight</td>
                        <td style="font-weight: 700; text-align: right;">{c_res['section_weight_kg_m']:.1f} kg/m</td>
                    </tr>
                    <tr>
                        <td style="color: #64748B; padding: 6px 0;">Combined Interaction Ratio</td>
                        <td style="font-weight: 700; text-align: right; color: {'#059669' if util <= 1.0 else '#DC2626'};">{util:.2f} &le; 1.00 [{c_res['status']}]</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown("#### Scaled Engineering Cross-Section")
            sec_dict = next((s for s in COLUMN_SECTIONS if s["name"] == c_res["recommended_section"]), COLUMN_SECTIONS[-1])
            fig_sec = create_section_geometry_fig(sec_dict)
            st.pyplot(fig_sec)

    st.markdown("---")
    st.plotly_chart(
        create_seismic_distribution_fig(n_stories, story_height, zone_name),
        use_container_width=True,
    )

# ==============================================================================
# TAB 2: BUILDING SCHEDULE & BIM VIEWS
# ==============================================================================
with tab2:
    schedule_data = predictor.generate_full_building_schedule(building_params)
    df_sched = schedule_data["schedule_table"]

    # Top KPI Metrics Strip
    kp1, kp2, kp3, kp4 = st.columns(4)
    with kp1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Total Structural Steel</div>
            <div class="kpi-value">{schedule_data['total_steel_tonnes']:.1f} <span style="font-size: 0.9rem; color: #64748B;">Tonnes</span></div>
            <div class="kpi-sub">G+{n_stories-1} Superstructure</div>
        </div>
        """, unsafe_allow_html=True)
    with kp2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Steel Intensity</div>
            <div class="kpi-value">{schedule_data['steel_intensity_kg_m2']:.1f} <span style="font-size: 0.9rem; color: #64748B;">kg/m²</span></div>
            <div class="kpi-sub">Total Area: {schedule_data['total_builtup_area_m2']:,.0f} m²</div>
        </div>
        """, unsafe_allow_html=True)
    with kp3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Estimated Cost (INR)</div>
            <div class="kpi-value" style="color: #0284C7;">₹ {schedule_data['estimated_cost_inr_lakhs']:.1f} <span style="font-size: 0.9rem; color: #64748B;">Lakhs</span></div>
            <div class="kpi-sub">@ ₹85 / kg fabricated</div>
        </div>
        """, unsafe_allow_html=True)
    with kp4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Estimated Cost (USD)</div>
            <div class="kpi-value" style="color: #0284C7;">$ {schedule_data['estimated_cost_usd']:,.0f}</div>
            <div class="kpi-sub">@ $1.05 / kg erected</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2D Elevation / 3D BIM Model switch
    col_vhead, col_vswitch = st.columns([2, 1.2])
    with col_vhead:
        st.markdown("### Structural Model & Elevation Preview")
    with col_vswitch:
        view_mode = st.radio("Display View", ["2D Elevation & Member Tags", "3D Interactive BIM Orbit"], horizontal=True)

    if "2D" in view_mode:
        fig_2d = create_2d_elevation_fig(n_stories, story_height, bay_x_count, bay_x_length, df_sched)
        st.plotly_chart(fig_2d, use_container_width=True)
    else:
        fig_3d = create_3d_frame_fig(n_stories, story_height, bay_x_count, bay_x_length, bay_y_count, bay_y_length)
        st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("### Complete Story-by-Story Sizing Schedule Table")
    st.dataframe(df_sched, use_container_width=True)

    col_chart, col_dl = st.columns([3, 1])
    with col_chart:
        st.plotly_chart(create_story_steel_takeoff_fig(df_sched), use_container_width=True)
    with col_dl:
        csv_bytes = df_sched.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export Schedule (CSV)",
            data=csv_bytes,
            file_name=f"steel_sizing_schedule_G+{n_stories-1}.csv",
            mime="text/csv",
        )

# ==============================================================================
# TAB 3: MODEL ANALYTICS & DIAGNOSTICS
# ==============================================================================
with tab3:
    st.markdown("### Machine Learning Engine Architecture & Verification")
    st.write("Supervised regression pipeline comparing Baseline Ridge, Random Forest, Gradient Boosting, and Multi-Layer Perceptron (MLP) Neural Networks.")

    metrics_path = "models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            all_m = json.load(f)

        st.markdown("#### Beam Sizing Models Benchmark (Target: Plastic Modulus Zp)")
        beam_m = all_m["beams"]["beam_zp_metrics"]
        b_df_rows = []
        for mname, mdata in beam_m.items():
            if isinstance(mdata, dict):
                b_df_rows.append({
                    "Model Architecture": mname.replace("_", " "),
                    "Test R² Score": mdata.get("test_r2"),
                    "RMSE (cm³)": mdata.get("test_rmse"),
                    "MAE (cm³)": mdata.get("test_mae"),
                    "MAPE (%)": f"{mdata.get('test_mape_percent'):.2f}%",
                    "5-Fold CV R²": f"{mdata.get('cv_r2_mean')} ± {mdata.get('cv_r2_std')}",
                })
        st.dataframe(pd.DataFrame(b_df_rows), use_container_width=True)

        st.markdown("#### Column Sizing Models Benchmark (Target: Required Area)")
        col_m = all_m["columns"]["column_area_metrics"]
        c_df_rows = []
        for mname, mdata in col_m.items():
            if isinstance(mdata, dict):
                c_df_rows.append({
                    "Model Architecture": mname.replace("_", " "),
                    "Test R² Score": mdata.get("test_r2"),
                    "RMSE (cm²)": mdata.get("test_rmse"),
                    "MAE (cm²)": mdata.get("test_mae"),
                    "MAPE (%)": f"{mdata.get('test_mape_percent'):.2f}%",
                    "5-Fold CV R²": f"{mdata.get('cv_r2_mean')} ± {mdata.get('cv_r2_std')}",
                })
        st.dataframe(pd.DataFrame(c_df_rows), use_container_width=True)

        st.markdown("#### Feature Importance Analysis (SHAP & Permutation Impact)")
        col_fi_b, col_fi_c = st.columns(2)
        with col_fi_b:
            b_fi = pd.DataFrame(beam_m.get("top_feature_importances", []))
            if not b_fi.empty:
                st.plotly_chart(create_feature_importance_fig(b_fi, "Key Drivers for Beam Capacity (Zp)"), use_container_width=True)
        with col_fi_c:
            c_fi = pd.DataFrame(col_m.get("top_feature_importances", []))
            if not c_fi.empty:
                st.plotly_chart(create_feature_importance_fig(c_fi, "Key Drivers for Column Area (A_req)"), use_container_width=True)

    else:
        st.info("Run python ml_pipeline.py to generate complete training benchmarks.")

# ==============================================================================
# TAB 4: ETABS / STAAD.PRO EXPORTER
# ==============================================================================
with tab4:
    st.markdown("### Parametric FEA CAD/BIM Exporter")
    st.write("Seamlessly export the preliminary ML structural design into ETABS (.e2k) or STAAD.Pro (.std) for final detailed FEA verification.")

    c_e2k, c_std = st.columns(2)
    with c_e2k:
        st.markdown("""
        <div class="pro-card">
            <div class="pro-card-header">CSI ETABS (.e2k Structural Text Model)</div>
            <p style="font-size: 0.85rem; color: #64748B;">Exports 3D grid lines, story heights, Fe250 steel materials, ISMB/ISHB frame sections, DL/LL uniform loads, and IS 1893 seismic load combinations.</p>
        </div>
        """, unsafe_allow_html=True)
        e2k_filename = f"SteelForm_G+{n_stories-1}_{zone_name.replace(' ', '_')}.e2k"
        e2k_path = generate_etabs_e2k(building_params, output_path=e2k_filename)
        with open(e2k_path, "r", encoding="utf-8") as f:
            e2k_content = f.read()
        st.download_button(
            label="Download ETABS (.e2k) File",
            data=e2k_content,
            file_name=e2k_filename,
            mime="text/plain",
        )

    with c_std:
        st.markdown("""
        <div class="pro-card">
            <div class="pro-card-header">Bentley STAAD.Pro (.std Command File)</div>
            <p style="font-size: 0.85rem; color: #64748B;">Exports 3D joint nodal coordinates, member incidences, IS 800:2007 LSD code parameters, and equivalent static seismic definitions.</p>
        </div>
        """, unsafe_allow_html=True)
        std_filename = f"SteelForm_G+{n_stories-1}_{zone_name.replace(' ', '_')}.std"
        std_path = generate_staad_std(building_params, output_path=std_filename)
        with open(std_path, "r", encoding="utf-8") as f:
            std_content = f.read()
        st.download_button(
            label="Download STAAD.Pro (.std) File",
            data=std_content,
            file_name=std_filename,
            mime="text/plain",
        )

    st.markdown("---")
    st.markdown("### ETABS OAPI Live Batch Automation Script")
    st.code(get_etabs_oapi_batch_template(), language="python")
