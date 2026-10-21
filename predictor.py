"""
Real-Time ML Inference & Structural Code Verification Engine
Provides instantaneous preliminary sizing for beams and columns with IS 800:2007 checks.
"""

from typing import Dict, List, Optional, Tuple
import os
import joblib
import pandas as pd

from steel_sections import (
    BEAM_SECTIONS,
    COLUMN_SECTIONS,
    find_economical_beam,
    find_economical_column,
    calculate_beam_moment_capacity,
    calculate_column_axial_capacity,
    FY,
    GAMMA_M0,
)

MODELS_DIR = "models"


class MemberSizingPredictor:
    """
    Inference interface that utilizes trained machine learning pipelines to predict
    economical steel sections for beams and columns in multistory frames.
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.beam_zp_model = None
        self.beam_weight_model = None
        self.col_area_model = None
        self.col_weight_model = None
        self._load_models()

    def _load_models(self):
        zp_path = os.path.join(self.models_dir, "beam_zp_model.joblib")
        bw_path = os.path.join(self.models_dir, "beam_weight_model.joblib")
        ca_path = os.path.join(self.models_dir, "column_area_model.joblib")
        cw_path = os.path.join(self.models_dir, "column_weight_model.joblib")

        if os.path.exists(zp_path):
            self.beam_zp_model = joblib.load(zp_path)
        if os.path.exists(bw_path):
            self.beam_weight_model = joblib.load(bw_path)
        if os.path.exists(ca_path):
            self.col_area_model = joblib.load(ca_path)
        if os.path.exists(cw_path):
            self.col_weight_model = joblib.load(cw_path)

    def predict_beam(self, params: Dict) -> Dict:
        """
        Predicts optimal ISMB/ISWB beam section for a specific story and location.
        """
        # Set default values for any missing parameters
        n_stories = params.get("n_stories", 10)
        story_num = params.get("story_num", 5)
        story_height = params.get("story_height", 3.5)
        bay_x_count = params.get("bay_x_count", 4)
        bay_y_count = params.get("bay_y_count", 3)
        bay_x_length = params.get("bay_x_length", 6.0)
        bay_y_length = params.get("bay_y_length", 6.0)
        span = params.get("span", bay_x_length)
        beam_location = params.get("beam_location", "Primary_Interior")
        dead_load = params.get("dead_load", 4.5)
        live_load = params.get("live_load", 3.0)
        seismic_zone = params.get("seismic_zone", "Zone IV")
        zone_z_map = {"Zone II": 0.10, "Zone III": 0.16, "Zone IV": 0.24, "Zone V": 0.36}
        zone_z = params.get("zone_z", zone_z_map.get(seismic_zone, 0.24))
        soil_type = params.get("soil_type", "Type II (Medium)")
        response_reduction_r = params.get("response_reduction_r", 5.0)
        importance_factor_i = params.get("importance_factor_i", 1.2)
        wind_speed = params.get("wind_speed", 44.0)

        input_df = pd.DataFrame([{
            "n_stories": n_stories,
            "story_num": story_num,
            "story_height": story_height,
            "bay_x_count": bay_x_count,
            "bay_y_count": bay_y_count,
            "bay_x_length": bay_x_length,
            "bay_y_length": bay_y_length,
            "span": span,
            "dead_load": dead_load,
            "live_load": live_load,
            "zone_z": zone_z,
            "response_reduction_r": response_reduction_r,
            "importance_factor_i": importance_factor_i,
            "wind_speed": wind_speed,
            "beam_location": beam_location,
            "seismic_zone": seismic_zone,
            "soil_type": soil_type,
        }])

        if self.beam_zp_model is not None:
            pred_zp = float(self.beam_zp_model.predict(input_df)[0])
            pred_weight = float(self.beam_weight_model.predict(input_df)[0]) if self.beam_weight_model else 50.0
        # Determine tributary width matching structural framing configuration
        if beam_location == "Primary_Interior":
            trib_w = bay_y_length
            is_secondary = False
        elif beam_location == "Primary_Exterior":
            trib_w = bay_y_length * 0.5
            is_secondary = False
        else:  # Secondary
            trib_w = bay_x_length * 0.33
            is_secondary = True

        # Factored gravity load and bending moment per IS 800:2007 Cl. 8.2.1.2
        wu = 1.5 * (dead_load * trib_w + live_load * trib_w + 0.6)  # kN/m
        mu_grav = (wu * (span ** 2)) / 10.0

        if not is_secondary:
            story_lat_ratio = max(0.0, (n_stories - story_num + 1) / n_stories)
            mu_sway_est = 20.0 * zone_z * story_lat_ratio * (story_height / 3.5)
            demand_mu = max(mu_grav, 1.2 * (mu_grav / 1.5 + mu_sway_est))
        else:
            demand_mu = mu_grav

        zp_req_physics = (demand_mu * 1e6 * GAMMA_M0) / (FY * 1e3)

        if self.beam_zp_model is not None:
            pred_zp = float(self.beam_zp_model.predict(input_df)[0])
            pred_weight = float(self.beam_weight_model.predict(input_df)[0]) if self.beam_weight_model else 50.0
            # Guardrail: section must always safely resist the factored design moment
            pred_zp = max(pred_zp, zp_req_physics)
        else:
            pred_zp = zp_req_physics
            pred_weight = 60.0

        pred_zp = max(50.0, pred_zp)
        optimal_sec = find_economical_beam(pred_zp)
        capacity_md = calculate_beam_moment_capacity(optimal_sec)

        utilization_ratio = round(demand_mu / max(capacity_md, 1e-3), 2)

        return {
            "predicted_zp_req_cm3": round(pred_zp, 1),
            "predicted_weight_kg_m": round(pred_weight, 1),
            "recommended_section": optimal_sec["name"],
            "section_depth_mm": optimal_sec["D"],
            "section_flange_width_mm": optimal_sec["Bf"],
            "section_weight_kg_m": optimal_sec["weight"],
            "section_zp_cm3": optimal_sec["Zpz"],
            "moment_capacity_kNm": round(capacity_md, 2),
            "factored_moment_demand_kNm": round(demand_mu, 2),
            "utilization_ratio": utilization_ratio,
            "status": "PASS" if utilization_ratio <= 1.0 else "CHECK",
        }

    def predict_column(self, params: Dict) -> Dict:
        """
        Predicts optimal ISHB / Built-up column section for a specific story and location.
        """
        n_stories = params.get("n_stories", 10)
        story_num = params.get("story_num", 1)  # Default base story
        story_height = params.get("story_height", 3.5)
        bay_x_count = params.get("bay_x_count", 4)
        bay_y_count = params.get("bay_y_count", 3)
        bay_x_length = params.get("bay_x_length", 6.0)
        bay_y_length = params.get("bay_y_length", 6.0)
        col_location = params.get("column_location", "Interior")
        dead_load = params.get("dead_load", 4.5)
        live_load = params.get("live_load", 3.0)
        seismic_zone = params.get("seismic_zone", "Zone IV")
        zone_z_map = {"Zone II": 0.10, "Zone III": 0.16, "Zone IV": 0.24, "Zone V": 0.36}
        zone_z = params.get("zone_z", zone_z_map.get(seismic_zone, 0.24))
        soil_type = params.get("soil_type", "Type II (Medium)")
        response_reduction_r = params.get("response_reduction_r", 5.0)
        importance_factor_i = params.get("importance_factor_i", 1.2)
        wind_speed = params.get("wind_speed", 44.0)

        # Tributary area calculation
        if col_location == "Corner":
            trib_area = (bay_x_length * 0.5) * (bay_y_length * 0.5)
        elif col_location == "Exterior":
            trib_area = bay_x_length * (bay_y_length * 0.5)
        else:
            trib_area = bay_x_length * bay_y_length

        input_df = pd.DataFrame([{
            "n_stories": n_stories,
            "story_num": story_num,
            "story_height": story_height,
            "bay_x_count": bay_x_count,
            "bay_y_count": bay_y_count,
            "bay_x_length": bay_x_length,
            "bay_y_length": bay_y_length,
            "trib_area_m2": trib_area,
            "dead_load": dead_load,
            "live_load": live_load,
            "zone_z": zone_z,
            "response_reduction_r": response_reduction_r,
            "importance_factor_i": importance_factor_i,
            "wind_speed": wind_speed,
            "column_location": col_location,
            "seismic_zone": seismic_zone,
            "soil_type": soil_type,
        }])

        remaining_stories = n_stories - story_num + 1
        pu_est = remaining_stories * ((dead_load + 0.8 * live_load + 0.8) * trib_area) * 1.5
        mu_est = pu_est * 0.03 + (15.0 if col_location != "Interior" else 5.0)

        if self.col_area_model is not None:
            pred_area = float(self.col_area_model.predict(input_df)[0])
            pred_weight = float(self.col_weight_model.predict(input_df)[0]) if self.col_weight_model else 80.0
        else:
            pred_area = max(35.0, (pu_est * 1000.0) / (150.0 * 100.0))
            pred_weight = 70.0

        pred_area = max(30.0, pred_area)
        L_eff = story_height * (1.0 if response_reduction_r == 5.0 else 0.85)

        optimal_sec = find_economical_column(
            A_req_cm2=pred_area,
            P_u_kN=pu_est,
            M_u_kNm=mu_est,
            L_eff_m=L_eff,
        )

        pd_capacity = calculate_column_axial_capacity(optimal_sec, L_eff_m=L_eff)
        md_capacity = calculate_beam_moment_capacity(optimal_sec)
        interaction_ratio = round((pu_est / max(pd_capacity, 1e-3)) + (mu_est / max(md_capacity, 1e-3)), 3)

        return {
            "predicted_area_req_cm2": round(pred_area, 1),
            "predicted_weight_kg_m": round(pred_weight, 1),
            "recommended_section": optimal_sec["name"],
            "section_depth_mm": optimal_sec["D"],
            "section_flange_width_mm": optimal_sec["Bf"],
            "section_weight_kg_m": optimal_sec["weight"],
            "section_area_cm2": optimal_sec["area"],
            "axial_capacity_kN": round(pd_capacity, 1),
            "factored_axial_demand_kN": round(pu_est, 1),
            "interaction_ratio": interaction_ratio,
            "status": "PASS" if interaction_ratio <= 1.0 else "CHECK",
        }

    def generate_full_building_schedule(self, params: Dict) -> Dict:
        """
        Generates the complete Story 1 to Story N member schedule for all beams and columns,
        along with steel tonnage, intensity (kg/m2), and structural cost estimates.
        """
        n_stories = params.get("n_stories", 10)
        bay_x_count = params.get("bay_x_count", 4)
        bay_y_count = params.get("bay_y_count", 3)
        bay_x_length = params.get("bay_x_length", 6.0)
        bay_y_length = params.get("bay_y_length", 6.0)
        story_height = params.get("story_height", 3.5)

        total_builtup_area = (bay_x_count * bay_x_length) * (bay_y_count * bay_y_length) * n_stories
        num_corner_cols = 4
        num_ext_cols = 2 * (bay_x_count - 1) + 2 * (bay_y_count - 1)
        num_int_cols = max(0, (bay_x_count - 1) * (bay_y_count - 1))
        
        num_beams_x = bay_x_count * (bay_y_count + 1)
        num_beams_y = bay_y_count * (bay_x_count + 1)

        schedule_rows = []
        total_steel_kg = 0.0

        for s in range(n_stories, 0, -1):  # From Top (Roof) down to Base
            s_params = dict(params)
            s_params["story_num"] = s

            # Predict Beams
            b_int = self.predict_beam({**s_params, "beam_location": "Primary_Interior", "span": bay_x_length})
            b_ext = self.predict_beam({**s_params, "beam_location": "Primary_Exterior", "span": bay_x_length})
            b_sec = self.predict_beam({**s_params, "beam_location": "Secondary", "span": bay_y_length})

            # Predict Columns
            c_int = self.predict_column({**s_params, "column_location": "Interior"})
            c_ext = self.predict_column({**s_params, "column_location": "Exterior"})
            c_crn = self.predict_column({**s_params, "column_location": "Corner"})

            # Steel takeoff for story
            beam_weight_story = (
                (num_beams_x * 0.5 * bay_x_length * b_int["section_weight_kg_m"]) +
                (num_beams_x * 0.5 * bay_x_length * b_ext["section_weight_kg_m"]) +
                (num_beams_y * bay_y_length * b_sec["section_weight_kg_m"])
            )
            col_weight_story = story_height * (
                (num_int_cols * c_int["section_weight_kg_m"]) +
                (num_ext_cols * c_ext["section_weight_kg_m"]) +
                (num_corner_cols * c_crn["section_weight_kg_m"])
            )

            story_total_kg = beam_weight_story + col_weight_story
            total_steel_kg += story_total_kg

            schedule_rows.append({
                "Story": f"Story {s}",
                "Elevation (m)": round(s * story_height, 1),
                "Beam (Int)": b_int["recommended_section"],
                "Beam (Ext)": b_ext["recommended_section"],
                "Beam (Sec)": b_sec["recommended_section"],
                "Column (Int)": c_int["recommended_section"],
                "Column (Ext)": c_ext["recommended_section"],
                "Column (Corner)": c_crn["recommended_section"],
                "Story Steel (Tonnes)": round(story_total_kg / 1000.0, 2),
            })

        df_schedule = pd.DataFrame(schedule_rows)
        total_tonnes = total_steel_kg / 1000.0
        steel_intensity = total_steel_kg / max(total_builtup_area, 1.0)  # kg/m2
        
        # Estimated fabricated steel cost (approx INR 85/kg or $1.05/kg fabricated & erected)
        est_cost_inr_lakhs = (total_steel_kg * 85.0) / 100000.0
        est_cost_usd = total_steel_kg * 1.05

        return {
            "schedule_table": df_schedule,
            "total_steel_tonnes": round(total_tonnes, 2),
            "total_builtup_area_m2": round(total_builtup_area, 1),
            "steel_intensity_kg_m2": round(steel_intensity, 2),
            "estimated_cost_inr_lakhs": round(est_cost_inr_lakhs, 2),
            "estimated_cost_usd": round(est_cost_usd, 0),
        }
