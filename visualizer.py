"""
Interactive 2D/3D Structural Engineering Visualizer & Drawing Engine
Provides high-fidelity, professional structural graphics:
1. 2D Multistory Frame Elevation with Color-Coded Member Sections
2. 3D Wireframe BIM-style Building Model
3. Engineering Cross-Section Dimension Drawing (ISMB I-Beam / ISHB Column / Box)
4. Dual-Panel IS 1893:2016 Seismic Demand Profiles (Diaphragm Forces Qi & Stepped Story Shear Vi)
5. Story Steel Takeoff & Breakdown Bar Charts
6. Model Feature Importance Analytics Visualizers
"""

from typing import Dict, List, Optional
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_2d_elevation_fig(
    n_stories: int,
    story_height: float,
    bay_x_count: int,
    bay_x_length: float,
    schedule_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Renders an interactive 2D Elevation drawing of the multistory steel frame.
    Annotates each story with recommended beam and column section designations.
    """
    fig = go.Figure()
    total_width = bay_x_count * bay_x_length
    total_height = n_stories * story_height

    # Build schedule mapping if available
    sched_map = {}
    if schedule_df is not None and not schedule_df.empty:
        for _, row in schedule_df.iterrows():
            s_name = str(row["Story"]).strip()
            sched_map[s_name] = {
                "b_int": row.get("Beam (Int)", "ISMB 400"),
                "b_ext": row.get("Beam (Ext)", "ISMB 350"),
                "c_int": row.get("Column (Int)", "ISHB 350"),
                "c_ext": row.get("Column (Ext)", "ISHB 250"),
                "c_crn": row.get("Column (Corner)", "ISHB 200"),
            }

    # Draw Columns
    for col_idx in range(bay_x_count + 1):
        x_pos = col_idx * bay_x_length
        is_ext = (col_idx == 0 or col_idx == bay_x_count)

        for s in range(1, n_stories + 1):
            y_bot = (s - 1) * story_height
            y_top = s * story_height
            s_name = f"Story {s}"
            
            # Determine section name & color
            sec_info = sched_map.get(s_name, {})
            sec_name = sec_info.get("c_ext" if is_ext else "c_int", "ISHB Column")
            
            # Color gradient: base heavy columns darker blue/purple, upper columns cyan
            color_val = f"rgba({max(20, 220 - s * 18)}, {min(220, 80 + s * 14)}, 230, 0.9)"
            line_width = max(3.0, 8.0 - (s * 0.4))

            fig.add_trace(go.Scatter(
                x=[x_pos, x_pos],
                y=[y_bot, y_top],
                mode="lines",
                line=dict(color=color_val, width=line_width),
                hoverinfo="text",
                hovertext=f"<b>Column Line {chr(65 + col_idx)}</b><br>{s_name} (Elev {y_bot:.1f}m - {y_top:.1f}m)<br>Role: {'Exterior' if is_ext else 'Interior'}<br><b>Section: {sec_name}</b>",
                showlegend=False,
            ))

    # Draw Beams
    for s in range(1, n_stories + 1):
        y_pos = s * story_height
        s_name = f"Story {s}"
        sec_info = sched_map.get(s_name, {})

        for bay_idx in range(bay_x_count):
            x_start = bay_idx * bay_x_length
            x_end = (bay_idx + 1) * bay_x_length
            is_ext_bay = (bay_idx == 0 or bay_idx == bay_x_count - 1)
            beam_sec = sec_info.get("b_ext" if is_ext_bay else "b_int", "ISMB Beam")

            fig.add_trace(go.Scatter(
                x=[x_start, x_end],
                y=[y_pos, y_pos],
                mode="lines",
                line=dict(color="#F97316", width=4.5),
                hoverinfo="text",
                hovertext=f"<b>{s_name} Floor Beam</b><br>Span: {bay_x_length:.1f}m (Bay {bay_idx+1})<br>Elevation: {y_pos:.1f}m<br><b>Section: {beam_sec}</b>",
                showlegend=False,
            ))

        # Add story label annotation
        int_sec = sec_info.get("b_int", "ISMB")
        fig.add_annotation(
            x=total_width * 0.5,
            y=y_pos + (story_height * 0.15),
            text=f"<b>{s_name}</b> | Beam: {int_sec}",
            showarrow=False,
            font=dict(size=10, color="#1F2937"),
            bgcolor="rgba(255, 255, 255, 0.75)",
            bordercolor="#D1D5DB",
            borderwidth=1,
        )

    # Base Support Symbols (Fixed Base)
    for col_idx in range(bay_x_count + 1):
        x_pos = col_idx * bay_x_length
        fig.add_trace(go.Scatter(
            x=[x_pos - 0.4, x_pos + 0.4],
            y=[0, 0],
            mode="lines",
            line=dict(color="#111827", width=5),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[x_pos],
            y=[-0.2],
            mode="markers",
            marker=dict(symbol="triangle-up", size=14, color="#374151"),
            hoverinfo="text",
            hovertext=f"Fixed Foundation Base (Grid {chr(65+col_idx)})",
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(
            text=f"<b>2D Multistory Frame Elevation (G+{n_stories-1}) with Member Sizing Schedule</b>",
            font=dict(size=16, color="#1E3A8A"),
        ),
        xaxis=dict(
            title="Building Span X (meters)",
            showgrid=True,
            gridcolor="#E5E7EB",
            zeroline=True,
            range=[-1.0, total_width + 1.0],
        ),
        yaxis=dict(
            title="Elevation (meters)",
            showgrid=True,
            gridcolor="#E5E7EB",
            zeroline=True,
            range=[-1.2, total_height + (story_height * 0.8)],
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=520,
        margin=dict(l=50, r=30, t=50, b=50),
    )
    return fig


def create_3d_frame_fig(
    n_stories: int,
    story_height: float,
    bay_x_count: int,
    bay_x_length: float,
    bay_y_count: int,
    bay_y_length: float,
) -> go.Figure:
    """
    Renders an interactive 3D wireframe BIM-style building visualization.
    """
    fig = go.Figure()

    # Column lines
    for i in range(bay_x_count + 1):
        x = i * bay_x_length
        for j in range(bay_y_count + 1):
            y = j * bay_y_length
            is_corner = (i in (0, bay_x_count) and j in (0, bay_y_count))
            col_color = "#3B82F6" if not is_corner else "#8B5CF6"

            fig.add_trace(go.Scatter3d(
                x=[x, x],
                y=[y, y],
                z=[0, n_stories * story_height],
                mode="lines",
                line=dict(color=col_color, width=4.5),
                hoverinfo="text",
                hovertext=f"Column Line ({chr(65+i)}, {j+1})<br>Height: {n_stories * story_height:.1f}m<br>Role: {'Corner' if is_corner else 'Typical'}",
                showlegend=False,
            ))

    # Floor Beams (X-direction & Y-direction)
    for s in range(1, n_stories + 1):
        z = s * story_height
        # X-beams
        for j in range(bay_y_count + 1):
            y = j * bay_y_length
            fig.add_trace(go.Scatter3d(
                x=[0, bay_x_count * bay_x_length],
                y=[y, y],
                z=[z, z],
                mode="lines",
                line=dict(color="#F97316", width=3),
                hoverinfo="text",
                hovertext=f"Story {s} Primary Beam (Grid Y={j+1})<br>Elevation: {z:.1f}m",
                showlegend=False,
            ))
        # Y-beams
        for i in range(bay_x_count + 1):
            x = i * bay_x_length
            fig.add_trace(go.Scatter3d(
                x=[x, x],
                y=[0, bay_y_count * bay_y_length],
                z=[z, z],
                mode="lines",
                line=dict(color="#10B981", width=3),
                hoverinfo="text",
                hovertext=f"Story {s} Secondary Beam (Grid X={chr(65+i)})<br>Elevation: {z:.1f}m",
                showlegend=False,
            ))

    # Base Slabs Wire
    fig.add_trace(go.Scatter3d(
        x=[0, bay_x_count * bay_x_length, bay_x_count * bay_x_length, 0, 0],
        y=[0, 0, bay_y_count * bay_y_length, bay_y_count * bay_y_length, 0],
        z=[0, 0, 0, 0, 0],
        mode="lines",
        line=dict(color="#111827", width=6),
        hoverinfo="text",
        hovertext="Base Ground Foundation",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>3D Interactive Structural Model (G+{n_stories-1} Frame)</b>",
            font=dict(size=16, color="#1E3A8A"),
        ),
        scene=dict(
            xaxis=dict(title="X Span (m)", backgroundcolor="#F9FAFB"),
            yaxis=dict(title="Y Span (m)", backgroundcolor="#F9FAFB"),
            zaxis=dict(title="Elevation Z (m)", backgroundcolor="#F9FAFB"),
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.2),
            ),
        ),
        height=540,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def create_section_geometry_fig(section: Dict) -> plt.Figure:
    """
    Renders an authentic, CAD-standard technical engineering cross-section drawing
    for hot-rolled ISMB beams, ISHB columns, plated columns, and built-up box sections.
    Includes authentic internal root fillets (R1), ANSI31 cross-hatching,
    offset witness lines, non-overlapping dimension callouts, and labeled neutral axes.
    """
    fig, ax = plt.subplots(figsize=(7.0, 6.5), dpi=140)
    
    name = section.get("name", "Steel Section")
    D = float(section.get("D", 225.0))  # mm
    Bf = float(section.get("Bf", 110.0))  # mm
    tw = float(section.get("tw", 6.5))  # mm
    tf = float(section.get("tf", 11.8))  # mm
    area = float(section.get("area", 39.7))  # cm2
    weight = float(section.get("weight", 31.2))  # kg/m
    zp = float(section.get("Zpz", 348.0))  # cm3
    izz = float(section.get("Izz", 3441.8))  # cm4
    iyy = float(section.get("Iyy", 218.3))  # cm4

    # Authentic root fillet radius per IS 808 / AISC standard
    R1 = max(6.5, min(16.0, 1.25 * tw))

    if "Box" in name:
        # Hollow Box Section with inner void
        outer_box = patches.Rectangle(
            (-Bf / 2, -D / 2), Bf, D,
            linewidth=2.0, edgecolor="#0F172A", facecolor="#E2E8F0", hatch="//", alpha=0.95
        )
        inner_box = patches.Rectangle(
            (-Bf / 2 + tw, -D / 2 + tf), Bf - 2 * tw, D - 2 * tf,
            linewidth=1.8, edgecolor="#0F172A", facecolor="#FFFFFF"
        )
        ax.add_patch(outer_box)
        ax.add_patch(inner_box)
        
        # Callout for wall thickness
        ax.annotate(
            f"Wall t = {tw:.1f} mm",
            xy=(-Bf / 2 + tw / 2, 0),
            xytext=(-Bf / 2 - 45, 10),
            arrowprops=dict(arrowstyle="->", color="#0F172A", lw=1.2, connectionstyle="angle,angleA=0,angleB=90,rad=0"),
            fontsize=9, fontweight="bold", color="#0F172A",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF", edgecolor="#CBD5E1")
        )
    else:
        # Authentic Hot-Rolled I-Section with 4 Circular Root Fillets (IS 808:1989)
        pts = []
        # 1. Top flange top edge (left to right)
        pts.append([-Bf / 2, D / 2])
        pts.append([Bf / 2, D / 2])
        # 2. Top-right flange tip
        pts.append([Bf / 2, D / 2 - tf])
        # 3. Top-right inner flange inward to fillet start
        pts.append([tw / 2 + R1, D / 2 - tf])
        # 4. Top-right root fillet arc down to web
        for a in np.linspace(np.pi / 2, np.pi, 12):
            pts.append([tw / 2 + R1 + R1 * np.cos(a), D / 2 - tf - R1 + R1 * np.sin(a)])
        # 5. Right web edge down
        pts.append([tw / 2, -D / 2 + tf + R1])
        # 6. Bottom-right root fillet arc out to bottom flange (center: tw/2 + R1, -D/2 + tf + R1)
        for a in np.linspace(np.pi, 3 * np.pi / 2, 12):
            pts.append([tw / 2 + R1 + R1 * np.cos(a), -D / 2 + tf + R1 + R1 * np.sin(a)])
        # 7. Bottom-right inner flange to tip
        pts.append([Bf / 2, -D / 2 + tf])
        pts.append([Bf / 2, -D / 2])
        # 8. Bottom flange bottom edge
        pts.append([-Bf / 2, -D / 2])
        pts.append([-Bf / 2, -D / 2 + tf])
        # 9. Bottom-left inner flange inward to fillet start
        pts.append([-tw / 2 - R1, -D / 2 + tf])
        # 10. Bottom-left root fillet arc up to web (center: -tw/2 - R1, -D/2 + tf + R1)
        for a in np.linspace(3 * np.pi / 2, 2 * np.pi, 12):
            pts.append([-tw / 2 - R1 + R1 * np.cos(a), -D / 2 + tf + R1 + R1 * np.sin(a)])
        # 11. Left web edge up
        pts.append([-tw / 2, D / 2 - tf - R1])
        # 12. Top-left root fillet arc out to top flange (center: -tw/2 - R1, D/2 - tf - R1)
        for a in np.linspace(0, np.pi / 2, 12):
            pts.append([-tw / 2 - R1 + R1 * np.cos(a), D / 2 - tf - R1 + R1 * np.sin(a)])
        # 13. Top-left inner flange to tip
        pts.append([-Bf / 2, D / 2 - tf])

        i_poly = patches.Polygon(
            pts, closed=True,
            facecolor="#E2E8F0", edgecolor="#0F172A",
            linewidth=2.0, hatch="//", alpha=0.95
        )
        ax.add_patch(i_poly)

        # Flange Thickness (tf) Leader Callout
        ax.annotate(
            f"tf = {tf:.1f} mm",
            xy=(-Bf / 2, D / 2 - tf / 2),
            xytext=(-Bf / 2 - 45, D / 2 + 10),
            arrowprops=dict(arrowstyle="->", color="#0F172A", lw=1.2, connectionstyle="angle,angleA=0,angleB=90,rad=0"),
            fontsize=9, fontweight="bold", color="#0F172A",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF", edgecolor="#CBD5E1")
        )

        # Web Thickness (tw) Leader Callout
        ax.annotate(
            f"tw = {tw:.1f} mm",
            xy=(-tw / 2, -D * 0.1),
            xytext=(-Bf / 2 - 45, -D * 0.1),
            arrowprops=dict(arrowstyle="->", color="#0F172A", lw=1.2),
            fontsize=9, fontweight="bold", color="#0F172A",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF", edgecolor="#CBD5E1")
        )

        # Root Fillet Radius (R1) Leader Callout
        ax.annotate(
            f"R1 = {R1:.1f} mm",
            xy=(tw / 2 + R1 * 0.28, D / 2 - tf - R1 * 0.32),
            xytext=(Bf / 2 + 35, D / 2 - tf - R1 - 8),
            arrowprops=dict(arrowstyle="->", color="#475569", lw=1.1, connectionstyle="arc3,rad=-0.15"),
            fontsize=8.5, color="#334155",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#FFFFFF", edgecolor="#CBD5E1")
        )

    # --- CAD WITNESS LINES & DIMENSIONS (RIGHT & TOP) ---
    # Depth D Dimension on the Right
    x_dim_d = (Bf / 2) + 26.0
    ax.plot([Bf / 2 + 3, x_dim_d + 6], [D / 2, D / 2], color="#94A3B8", lw=0.9, linestyle="-")
    ax.plot([Bf / 2 + 3, x_dim_d + 6], [-D / 2, -D / 2], color="#94A3B8", lw=0.9, linestyle="-")
    ax.annotate(
        "", xy=(x_dim_d, -D / 2), xytext=(x_dim_d, D / 2),
        arrowprops=dict(arrowstyle="<->", color="#0F172A", lw=1.3, shrinkA=0, shrinkB=0)
    )
    # Dimension text centered with white background box (prevents line clashing)
    ax.text(
        x_dim_d, 0, f"D = {D:.0f} mm",
        ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="#0F172A",
        bbox=dict(boxstyle="square,pad=0.28", facecolor="#FFFFFF", edgecolor="none")
    )

    # Flange Width Bf Dimension on Top
    y_dim_b = (D / 2) + 22.0
    ax.plot([-Bf / 2, -Bf / 2], [D / 2 + 2, y_dim_b + 5], color="#94A3B8", lw=0.9, linestyle="-")
    ax.plot([Bf / 2, Bf / 2], [D / 2 + 2, y_dim_b + 5], color="#94A3B8", lw=0.9, linestyle="-")
    ax.annotate(
        "", xy=(-Bf / 2, y_dim_b), xytext=(Bf / 2, y_dim_b),
        arrowprops=dict(arrowstyle="<->", color="#0F172A", lw=1.3, shrinkA=0, shrinkB=0)
    )
    # Dimension text centered with white background box
    ax.text(
        0, y_dim_b, f"Bf = {Bf:.0f} mm",
        ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="#0F172A",
        bbox=dict(boxstyle="square,pad=0.28", facecolor="#FFFFFF", edgecolor="none")
    )

    # --- CENTERLINES / NEUTRAL AXES (WITH END BADGES) ---
    axis_span_z = Bf / 2 + 65.0
    axis_span_y = D / 2 + 38.0

    # Major Neutral Axis Z-Z (Horizontal)
    ax.plot([-axis_span_z, axis_span_z], [0, 0], color="#DC2626", linestyle="-.", linewidth=1.1, alpha=0.85)
    for x_end in [-axis_span_z, axis_span_z]:
        ax.text(
            x_end, 0, "Z", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#DC2626",
            bbox=dict(boxstyle="circle,pad=0.22", facecolor="#FEE2E2", edgecolor="#DC2626", lw=0.8)
        )

    # Minor Neutral Axis Y-Y (Vertical)
    ax.plot([0, 0], [-axis_span_y, axis_span_y], color="#2563EB", linestyle=":", linewidth=1.1, alpha=0.85)
    for y_end in [-axis_span_y, axis_span_y]:
        ax.text(
            0, y_end, "Y", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#2563EB",
            bbox=dict(boxstyle="circle,pad=0.22", facecolor="#DBEAFE", edgecolor="#2563EB", lw=0.8)
        )

    # Limits and Margins
    x_max = Bf / 2 + 75.0
    y_max = D / 2 + 60.0
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max - 20, y_max + 20)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title & CAD Information Header
    ax.text(
        0, y_max + 14, f"HOT-ROLLED STEEL SECTION PROFILE — {name}",
        ha="center", va="bottom",
        fontsize=11.0, fontweight="bold", color="#0F172A"
    )
    ax.text(
        0, y_max + 2, "Conforms to IS 808:1989 & IS 800:2007 Limit State Design",
        ha="center", va="bottom",
        fontsize=8.2, color="#64748B"
    )

    # Technical Data Schedule Strip below profile
    meta_text = (
        f"Weight: {weight:.1f} kg/m   |   Area: {area:.1f} cm²   |   Zpz: {zp:.0f} cm³   |   "
        f"Izz: {izz:.0f} cm⁴   |   Iyy: {iyy:.0f} cm⁴"
    )
    ax.text(
        0, -y_max - 12, meta_text,
        ha="center", va="top",
        fontsize=7.8, color="#334155", family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#E2E8F0", lw=1.0)
    )

    plt.tight_layout()
    return fig


def create_seismic_distribution_fig(
    n_stories: int,
    story_height: float,
    zone_name: str,
    base_shear_kn: float = 1200.0,
) -> go.Figure:
    """
    Renders high-fidelity dual-panel structural engineering diagrams for IS 1893:2016:
    Panel 1: Lateral Floor Diaphragm Force Qi (horizontal load distribution at floor slabs)
    Panel 2: Stepped Cumulative Story Shear Vi envelope from roof down to base shear Vb
    """
    levels = list(range(1, n_stories + 1))
    elevations = [s * story_height for s in levels]
    
    # IS 1893:2016 Cl. 7.6.3 Parabolic distribution: Qi = Vb * (Wi * hi^2) / sum(Wj * hj^2)
    # Realistic seismic weights: Typical floor has full DL + partition + 25% LL;
    # Roof has DL + equipment/parapet without live load.
    w_floor = 450.0  # kN
    w_roof = 380.0   # kN
    weights = [w_floor] * (n_stories - 1) + [w_roof]
    
    w_h2 = [w * (h ** 2) for w, h in zip(weights, elevations)]
    sum_wh2 = sum(w_h2)
    
    # Calculate realistic floor lateral forces Qi
    q_forces = [base_shear_kn * (val / sum_wh2) for val in w_h2]
    
    # Cumulative Story Shear Vi at each story (from roof downward)
    v_shears = [sum(q_forces[i:]) for i in range(n_stories)]
    
    # Build stepped shear coordinates for true structural shear diagram
    # Story i spans from (i-1)*h to i*h with constant shear v_shears[i-1]
    step_y = [0.0]
    step_x = [v_shears[0]]  # Base shear at ground
    
    for i in range(n_stories):
        h_bot = i * story_height
        h_top = (i + 1) * story_height
        v_val = v_shears[i]
        
        step_y.extend([h_bot, h_top])
        step_x.extend([v_val, v_val])

    # Create 2 synchronized subplots side-by-side
    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        subplot_titles=(
            "<b>Floor Lateral Demand (Q<sub>i</sub>)</b>",
            "<b>Cumulative Story Shear Diagram (V<sub>i</sub>)</b>"
        ),
        horizontal_spacing=0.12,
    )

    # --- SUBPLOT 1: Lateral Diaphragm Forces Qi ---
    # Horizontal diaphragm stems from center line
    for q_val, elev, s_num in zip(q_forces, elevations, levels):
        fig.add_trace(go.Scatter(
            x=[0, q_val],
            y=[elev, elev],
            mode="lines",
            line=dict(color="#94A3B8", width=1.5, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        ), row=1, col=1)

    # Parabolic floor force marker curve
    fig.add_trace(go.Scatter(
        x=q_forces,
        y=elevations,
        mode="lines+markers+text",
        name="Floor Force Q<sub>i</sub>",
        line=dict(color="#0284C7", width=3, shape="spline"),
        marker=dict(size=8, color="#0284C7", symbol="circle", line=dict(color="#FFFFFF", width=1.5)),
        text=[f"{q:.0f} kN" for q in q_forces],
        textposition="top right",
        textfont=dict(size=9.5, color="#0F172A"),
        hoverinfo="text",
        hovertext=[f"<b>Story {s} (Elev {h:.1f}m)</b><br>Lateral Floor Force: {q:.1f} kN" for s, h, q in zip(levels, elevations, q_forces)],
        showlegend=False,
    ), row=1, col=1)

    # --- SUBPLOT 2: Cumulative Story Shear Diagram Vi ---
    # Stepped area fill
    fig.add_trace(go.Scatter(
        x=step_x,
        y=step_y,
        mode="lines",
        name="Story Shear V<sub>i</sub>",
        fill="tozerox",
        fillcolor="rgba(37, 99, 235, 0.12)",
        line=dict(color="#2563EB", width=2.8),
        hoverinfo="text",
        hovertext=[f"Story Shear Envelope: {v:.1f} kN at {y:.1f}m" for v, y in zip(step_x, step_y)],
        showlegend=False,
    ), row=1, col=2)

    # Mark story levels on shear diagram
    fig.add_trace(go.Scatter(
        x=v_shears,
        y=elevations,
        mode="markers",
        marker=dict(size=6, color="#1D4ED8", symbol="diamond"),
        hoverinfo="text",
        hovertext=[f"<b>Story {s} Shear:</b> {v:.1f} kN" for s, v in zip(levels, v_shears)],
        showlegend=False,
    ), row=1, col=2)

    # Annotate Base Shear Vb
    fig.add_annotation(
        xref="x2", yref="y2",
        x=v_shears[0] * 0.95, y=0.5,
        text=f"<b>Base Shear V<sub>B</sub> = {v_shears[0]:.0f} kN</b>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowcolor="#1E3A8A",
        font=dict(size=10.5, color="#1E3A8A"),
        bgcolor="#EFF6FF",
        bordercolor="#BFDBFE",
        borderwidth=1,
        row=1, col=2,
    )

    fig.update_layout(
        title=dict(
            text=f"<b>IS 1893:2016 Seismic Design Force & Story Shear Profiles ({zone_name})</b>",
            font=dict(size=14.5, color="#0F172A"),
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=400,
        margin=dict(l=50, r=40, t=55, b=45),
    )

    fig.update_xaxes(
        title_text="Diaphragm Force Q<sub>i</sub> (kN)",
        showgrid=True, gridcolor="#F1F5F9",
        zeroline=True, zerolinecolor="#CBD5E1",
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Cumulative Story Shear V<sub>i</sub> (kN)",
        showgrid=True, gridcolor="#F1F5F9",
        zeroline=True, zerolinecolor="#CBD5E1",
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="Height Elevation (meters)",
        showgrid=True, gridcolor="#F1F5F9",
        range=[-0.5, (n_stories * story_height) + (story_height * 0.5)],
        row=1, col=1
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#F1F5F9",
        range=[-0.5, (n_stories * story_height) + (story_height * 0.5)],
        row=1, col=2
    )

    return fig


def create_story_steel_takeoff_fig(df_sched: pd.DataFrame) -> go.Figure:
    """
    Renders a modern, professional steel quantity takeoff bar chart per story.
    """
    fig = go.Figure()

    stories = df_sched["Story"].tolist()
    tonnes = df_sched["Story Steel (Tonnes)"].tolist()
    elevations = df_sched["Elevation (m)"].tolist()
    b_int = df_sched["Beam (Int)"].tolist()
    c_int = df_sched["Column (Int)"].tolist()

    avg_tonnes = float(np.mean(tonnes))

    # Gradient steel bars
    fig.add_trace(go.Bar(
        y=stories,
        x=tonnes,
        orientation="h",
        marker=dict(
            color=tonnes,
            colorscale=[[0, "#38BDF8"], [0.5, "#0284C7"], [1.0, "#1E3A8A"]],
            showscale=False,
            line=dict(color="#0F172A", width=1),
        ),
        text=[f"<b>{t:.1f} T</b>" for t in tonnes],
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(color="#FFFFFF", size=11, family="Inter"),
        hoverinfo="text",
        hovertext=[
            f"<b>{s} (Elev {el:.1f}m)</b><br>Steel Tonnage: <b>{t:.2f} Tonnes</b><br>Primary Beam: {bi}<br>Interior Column: {ci}"
            for s, el, t, bi, ci in zip(stories, elevations, tonnes, b_int, c_int)
        ],
    ))

    # Benchmark average vertical line
    fig.add_vline(
        x=avg_tonnes,
        line_width=2,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Mean: {avg_tonnes:.1f} T",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#DC2626"),
    )

    fig.update_layout(
        title=dict(
            text="<b>Structural Steel Takeoff by Story Level (Tonnes)</b>",
            font=dict(size=14.5, color="#0F172A"),
        ),
        xaxis=dict(
            title="Fabricated Steel Weight (Tonnes)",
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        yaxis=dict(
            title="",
            autorange="reversed",  # Story 10 on top, Story 1 at bottom
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=380,
        margin=dict(l=60, r=40, t=50, b=40),
    )
    return fig


def create_feature_importance_fig(fi_df: pd.DataFrame, title: str) -> go.Figure:
    """
    Renders clean, styled horizontal bar chart for machine learning feature importances.
    """
    fig = go.Figure()

    if fi_df.empty:
        return fig

    # Human-readable feature names mapping
    feat_name_map = {
        "span": "Beam Span Length",
        "dead_load": "Dead Load (DL)",
        "live_load": "Live Load (LL)",
        "trib_area_m2": "Column Tributary Area",
        "story_num": "Story Elevation Level",
        "n_stories": "Total Building Height",
        "story_height": "Story Floor Height",
        "zone_z": "Seismic Zone Factor (Z)",
        "response_reduction_r": "Response Reduction (R)",
        "importance_factor_i": "Importance Factor (I)",
        "wind_speed": "Basic Wind Speed (Vb)",
        "bay_x_length": "Bay X Dimension",
        "bay_y_length": "Bay Y Dimension",
        "bay_x_count": "Bay X Count",
        "bay_y_count": "Bay Y Count",
    }

    df_plot = fi_df.copy()
    df_plot["clean_feature"] = df_plot["feature"].apply(
        lambda f: feat_name_map.get(f, f.replace("cat__", "").replace("_", " ").title())
    )
    df_plot = df_plot.sort_values(by="importance", ascending=True)

    fig.add_trace(go.Bar(
        y=df_plot["clean_feature"],
        x=df_plot["importance"] * 100.0,
        orientation="h",
        marker=dict(
            color=df_plot["importance"],
            colorscale=[[0, "#BAE6FD"], [1.0, "#0284C7"]],
            showscale=False,
            line=dict(color="#0284C7", width=1),
        ),
        text=[f"<b>{val * 100:.1f}%</b>" for val in df_plot["importance"]],
        textposition="outside",
        textfont=dict(size=10, color="#0F172A"),
        hoverinfo="text",
        hovertext=[f"<b>{f}</b><br>Relative Model Influence: {v*100:.1f}%" for f, v in zip(df_plot["clean_feature"], df_plot["importance"])],
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=13.5, color="#0F172A"),
        ),
        xaxis=dict(
            title="Relative Importance Weight (%)",
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        yaxis=dict(title=""),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=320,
        margin=dict(l=150, r=40, t=40, b=40),
    )
    return fig
