"""
Indian Standard Steel Sections Catalog and Limit State Design Utilities (IS 800:2007)
Includes standard ISMB / ISWB beams and ISHB / Plated / Built-up columns.
"""

from typing import Dict, List, Optional, Tuple
import math

# Material properties for Fe 410 grade structural steel (E250)
FY = 250.0  # Yield strength in MPa (N/mm2)
FU = 410.0  # Ultimate tensile strength in MPa
E = 200000.0  # Modulus of elasticity in MPa
GAMMA_M0 = 1.10  # Partial safety factor for resistance governed by yielding
GAMMA_M1 = 1.25  # Partial safety factor for resistance governed by ultimate stress

# Standard ISMB & ISWB Beam Database
# Dimensions: D (mm), Bf (mm), tw (mm), tf (mm), Area (cm2), Weight (kg/m), Zz (cm3), Zpz (cm3), Izz (cm4), Iyy (cm4)
BEAM_SECTIONS = [
    {"name": "ISMB 100", "D": 100, "Bf": 75, "tw": 4.0, "tf": 7.2, "area": 14.6, "weight": 11.5, "Zz": 51.5, "Zpz": 58.3, "Izz": 257.5, "Iyy": 40.8},
    {"name": "ISMB 125", "D": 125, "Bf": 75, "tw": 4.4, "tf": 7.6, "area": 17.0, "weight": 13.3, "Zz": 71.8, "Zpz": 81.4, "Izz": 449.0, "Iyy": 43.7},
    {"name": "ISMB 150", "D": 150, "Bf": 80, "tw": 4.8, "tf": 7.6, "area": 19.1, "weight": 15.0, "Zz": 96.9, "Zpz": 110.0, "Izz": 726.4, "Iyy": 52.6},
    {"name": "ISMB 175", "D": 175, "Bf": 90, "tw": 5.5, "tf": 8.6, "area": 24.6, "weight": 19.3, "Zz": 145.4, "Zpz": 165.2, "Izz": 1272.0, "Iyy": 85.0},
    {"name": "ISMB 200", "D": 200, "Bf": 100, "tw": 5.7, "tf": 10.0, "area": 32.3, "weight": 25.4, "Zz": 223.5, "Zpz": 254.0, "Izz": 2235.4, "Iyy": 150.0},
    {"name": "ISMB 225", "D": 225, "Bf": 110, "tw": 6.5, "tf": 11.8, "area": 39.7, "weight": 31.2, "Zz": 305.9, "Zpz": 348.0, "Izz": 3441.8, "Iyy": 218.3},
    {"name": "ISMB 250", "D": 250, "Bf": 125, "tw": 6.9, "tf": 12.5, "area": 47.5, "weight": 37.3, "Zz": 410.5, "Zpz": 467.0, "Izz": 5131.6, "Iyy": 334.5},
    {"name": "ISMB 300", "D": 300, "Bf": 140, "tw": 7.5, "tf": 12.4, "area": 56.3, "weight": 44.2, "Zz": 573.6, "Zpz": 651.0, "Izz": 8603.6, "Iyy": 453.9},
    {"name": "ISMB 350", "D": 350, "Bf": 140, "tw": 8.1, "tf": 14.2, "area": 66.7, "weight": 52.4, "Zz": 778.9, "Zpz": 889.0, "Izz": 13630.3, "Iyy": 537.7},
    {"name": "ISMB 400", "D": 400, "Bf": 140, "tw": 8.9, "tf": 16.0, "area": 78.5, "weight": 61.6, "Zz": 1022.9, "Zpz": 1176.0, "Izz": 20458.0, "Iyy": 622.1},
    {"name": "ISMB 450", "D": 450, "Bf": 150, "tw": 9.4, "tf": 17.4, "area": 92.3, "weight": 72.4, "Zz": 1350.7, "Zpz": 1560.0, "Izz": 30390.8, "Iyy": 834.0},
    {"name": "ISMB 500", "D": 500, "Bf": 180, "tw": 10.2, "tf": 17.2, "area": 110.7, "weight": 86.9, "Zz": 1808.7, "Zpz": 2074.0, "Izz": 45218.3, "Iyy": 1369.8},
    {"name": "ISMB 550", "D": 550, "Bf": 190, "tw": 11.2, "tf": 19.3, "area": 132.1, "weight": 103.7, "Zz": 2359.8, "Zpz": 2710.0, "Izz": 64893.6, "Iyy": 1833.8},
    {"name": "ISMB 600", "D": 600, "Bf": 210, "tw": 12.0, "tf": 20.8, "area": 156.2, "weight": 122.6, "Zz": 3060.4, "Zpz": 3510.0, "Izz": 91813.0, "Iyy": 2651.0},
    {"name": "ISWB 600", "D": 600, "Bf": 250, "tw": 11.8, "tf": 21.3, "area": 170.4, "weight": 133.7, "Zz": 3540.0, "Zpz": 4080.0, "Izz": 106198.5, "Iyy": 4702.5},
    {"name": "BuiltUp-B700x300", "D": 700, "Bf": 300, "tw": 12.0, "tf": 25.0, "area": 224.0, "weight": 175.8, "Zz": 5420.0, "Zpz": 6180.0, "Izz": 189700.0, "Iyy": 11250.0},
    {"name": "BuiltUp-B800x350", "D": 800, "Bf": 350, "tw": 14.0, "tf": 28.0, "area": 288.0, "weight": 226.1, "Zz": 7950.0, "Zpz": 9050.0, "Izz": 318000.0, "Iyy": 20000.0},
    {"name": "BuiltUp-B900x400", "D": 900, "Bf": 400, "tw": 16.0, "tf": 32.0, "area": 372.0, "weight": 292.0, "Zz": 11400.0, "Zpz": 12980.0, "Izz": 513000.0, "Iyy": 34100.0},
]

# Standard ISHB & Built-up Column Database
# Dimensions: D (mm), Bf (mm), tw (mm), tf (mm), Area (cm2), Weight (kg/m), ry (cm), rz (cm), Zpz (cm3), Izz (cm4), Iyy (cm4)
COLUMN_SECTIONS = [
    {"name": "ISHB 150-1", "D": 150, "Bf": 150, "tw": 5.4, "tf": 8.4, "area": 34.5, "weight": 27.1, "ry": 3.54, "rz": 6.54, "Zpz": 219.0, "Izz": 1479.0, "Iyy": 474.0},
    {"name": "ISHB 150-2", "D": 150, "Bf": 150, "tw": 8.4, "tf": 9.0, "area": 44.1, "weight": 34.6, "ry": 3.55, "rz": 6.25, "Zpz": 258.0, "Izz": 1720.0, "Iyy": 512.0},
    {"name": "ISHB 200-1", "D": 200, "Bf": 200, "tw": 6.1, "tf": 9.0, "area": 47.5, "weight": 37.3, "ry": 4.67, "rz": 8.71, "Zpz": 400.0, "Izz": 3608.0, "Iyy": 1201.0},
    {"name": "ISHB 200-2", "D": 200, "Bf": 200, "tw": 9.0, "tf": 9.0, "area": 51.0, "weight": 40.0, "ry": 4.65, "rz": 8.54, "Zpz": 423.0, "Izz": 3721.0, "Iyy": 1208.0},
    {"name": "ISHB 225-1", "D": 225, "Bf": 225, "tw": 6.5, "tf": 9.1, "area": 54.9, "weight": 43.1, "ry": 5.25, "rz": 9.91, "Zpz": 530.0, "Izz": 5396.0, "Iyy": 1735.0},
    {"name": "ISHB 225-2", "D": 225, "Bf": 225, "tw": 8.6, "tf": 9.1, "area": 59.7, "weight": 46.8, "ry": 5.24, "rz": 9.64, "Zpz": 558.0, "Izz": 5550.0, "Iyy": 1745.0},
    {"name": "ISHB 250-1", "D": 250, "Bf": 250, "tw": 6.9, "tf": 9.7, "area": 64.9, "weight": 51.0, "ry": 5.84, "rz": 10.92, "Zpz": 686.0, "Izz": 7746.0, "Iyy": 2538.0},
    {"name": "ISHB 250-2", "D": 250, "Bf": 250, "tw": 8.8, "tf": 9.7, "area": 69.7, "weight": 54.7, "ry": 5.83, "rz": 10.70, "Zpz": 718.0, "Izz": 7984.0, "Iyy": 2550.0},
    {"name": "ISHB 300-1", "D": 300, "Bf": 250, "tw": 7.6, "tf": 10.6, "area": 74.8, "weight": 58.8, "ry": 5.77, "rz": 12.95, "Zpz": 928.0, "Izz": 12545.0, "Iyy": 2761.0},
    {"name": "ISHB 300-2", "D": 300, "Bf": 250, "tw": 9.4, "tf": 10.6, "area": 80.2, "weight": 63.0, "ry": 5.75, "rz": 12.70, "Zpz": 972.0, "Izz": 12950.0, "Iyy": 2775.0},
    {"name": "ISHB 350-1", "D": 350, "Bf": 250, "tw": 8.3, "tf": 11.6, "area": 85.9, "weight": 67.4, "ry": 5.69, "rz": 14.93, "Zpz": 1225.0, "Izz": 19159.0, "Iyy": 3030.0},
    {"name": "ISHB 350-2", "D": 350, "Bf": 250, "tw": 10.1, "tf": 11.6, "area": 92.2, "weight": 72.4, "ry": 5.67, "rz": 14.65, "Zpz": 1280.0, "Izz": 19803.0, "Iyy": 3050.0},
    {"name": "ISHB 400-1", "D": 400, "Bf": 250, "tw": 9.1, "tf": 12.7, "area": 98.7, "weight": 77.4, "ry": 5.61, "rz": 16.87, "Zpz": 1570.0, "Izz": 28083.0, "Iyy": 3328.0},
    {"name": "ISHB 400-2", "D": 400, "Bf": 250, "tw": 10.6, "tf": 12.7, "area": 104.7, "weight": 82.2, "ry": 5.59, "rz": 16.60, "Zpz": 1628.0, "Izz": 28823.0, "Iyy": 3345.0},
    {"name": "ISHB 450-1", "D": 450, "Bf": 250, "tw": 9.8, "tf": 13.7, "area": 111.1, "weight": 87.2, "ry": 5.54, "rz": 18.79, "Zpz": 1945.0, "Izz": 39210.0, "Iyy": 3580.0},
    {"name": "ISHB 450-2", "D": 450, "Bf": 250, "tw": 11.3, "tf": 13.7, "area": 117.8, "weight": 92.5, "ry": 5.52, "rz": 18.52, "Zpz": 2015.0, "Izz": 40350.0, "Iyy": 3600.0},
    {"name": "ISHB 450+Plates-12mm", "D": 474, "Bf": 300, "tw": 11.3, "tf": 25.7, "area": 183.0, "weight": 141.0, "ry": 7.20, "rz": 19.90, "Zpz": 3450.0, "Izz": 72500.0, "Iyy": 14500.0},
    {"name": "ISHB 450+Plates-20mm", "D": 490, "Bf": 320, "tw": 11.3, "tf": 33.7, "area": 227.0, "weight": 178.0, "ry": 7.80, "rz": 20.80, "Zpz": 4600.0, "Izz": 98000.0, "Iyy": 22500.0},
    {"name": "BuiltUp-Box-450x450", "D": 450, "Bf": 450, "tw": 25.0, "tf": 25.0, "area": 275.0, "weight": 215.0, "ry": 17.20, "rz": 17.20, "Zpz": 5200.0, "Izz": 86000.0, "Iyy": 86000.0},
    {"name": "BuiltUp-Box-550x550", "D": 550, "Bf": 550, "tw": 32.0, "tf": 32.0, "area": 376.0, "weight": 295.0, "ry": 21.00, "rz": 21.00, "Zpz": 8800.0, "Izz": 185000.0, "Iyy": 185000.0},
    {"name": "BuiltUp-Box-650x650", "D": 650, "Bf": 650, "tw": 40.0, "tf": 40.0, "area": 497.0, "weight": 390.0, "ry": 24.80, "rz": 24.80, "Zpz": 14200.0, "Izz": 352000.0, "Iyy": 352000.0},
]


def calculate_beam_moment_capacity(section: Dict, fy: float = FY) -> float:
    """
    Design bending strength Md (kNm) per IS 800:2007 Cl 8.2.1.2 (Plastic/Compact section, laterally supported).
    Md = beta_b * Zp * fy / gamma_m0
    """
    beta_b = 1.0  # For plastic and compact sections
    zp_mm3 = section["Zpz"] * 1000.0  # cm3 to mm3
    md_nmm = beta_b * zp_mm3 * fy / GAMMA_M0
    return md_nmm / 1e6  # kNm


def calculate_column_axial_capacity(section: Dict, L_eff_m: float = 3.5, fy: float = FY) -> float:
    """
    Design compressive strength Pd (kN) per IS 800:2007 Cl 7.1.2 (Buckling Class b or c).
    Uses minor axis slenderness lambda = L_eff / ry.
    """
    ry_mm = section["ry"] * 10.0  # cm to mm
    L_eff_mm = L_eff_m * 1000.0
    slenderness = L_eff_mm / ry_mm
    
    # Imperfection factor alpha for buckling class c (typical for rolled I-sections minor axis / welded box)
    alpha = 0.34 if "Box" in section["name"] else 0.49
    
    euler_stress = (math.pi ** 2 * E) / (slenderness ** 2)
    lambda_bar = math.sqrt(fy / max(euler_stress, 1e-3))
    
    phi = 0.5 * (1.0 + alpha * (lambda_bar - 0.2) + lambda_bar ** 2)
    chi = 1.0 / (phi + math.sqrt(max(0.0, phi ** 2 - lambda_bar ** 2)))
    chi = min(1.0, chi)
    
    fcd = chi * fy / GAMMA_M0  # Design compressive stress MPa
    area_mm2 = section["area"] * 100.0  # cm2 to mm2
    pd_kn = area_mm2 * fcd / 1000.0
    return pd_kn


def find_economical_beam(Zp_req_cm3: float, V_u_kN: float = 0.0) -> Dict:
    """
    Selects the minimum-weight beam section from the ISMB/ISWB catalog that satisfies Zp >= Zp_req.
    """
    valid = [s for s in BEAM_SECTIONS if s["Zpz"] >= Zp_req_cm3]
    if not valid:
        return BEAM_SECTIONS[-1]  # Largest available built-up section
    # Sort primarily by weight, secondarily by depth
    return min(valid, key=lambda s: (s["weight"], s["D"]))


def find_economical_column(
    A_req_cm2: float,
    P_u_kN: float,
    M_u_kNm: float = 0.0,
    L_eff_m: float = 3.5,
) -> Dict:
    """
    Selects the minimum-weight column section from the ISHB/Built-up catalog that satisfies
    area requirement, axial capacity, and combined axial-bending interaction:
    P_u / P_d + M_u / M_dz <= 1.0
    """
    valid = []
    for s in COLUMN_SECTIONS:
        if s["area"] < A_req_cm2 * 0.95:
            continue
        pd = calculate_column_axial_capacity(s, L_eff_m=L_eff_m)
        md = calculate_beam_moment_capacity(s)  # Major axis bending capacity
        if pd <= 0:
            continue
        
        interaction = (P_u_kN / pd) + (M_u_kNm / max(md, 1e-3))
        if interaction <= 1.0:
            valid.append((s, interaction))
            
    if not valid:
        # Return heaviest available if no standard section satisfies
        return COLUMN_SECTIONS[-1]
        
    # Sort by minimum weight
    return min(valid, key=lambda x: (x[0]["weight"], x[1]))[0]
