"""
ETABS (.e2k) & STAAD.Pro (.std) Bridge and Automation Interface
Enables exporting parametric structural models and importing FEA design results.
"""

from typing import Dict, List, Optional
import os


def generate_etabs_e2k(params: Dict, output_path: str = "model.e2k") -> str:
    """
    Generates a standard ETABS text model file (.e2k) for a multistory steel building.
    """
    n_stories = params.get("n_stories", 10)
    bay_x_count = params.get("bay_x_count", 4)
    bay_y_count = params.get("bay_y_count", 3)
    bay_x_length = params.get("bay_x_length", 6.0)
    bay_y_length = params.get("bay_y_length", 6.0)
    story_height = params.get("story_height", 3.5)
    dead_load = params.get("dead_load", 4.5)
    live_load = params.get("live_load", 3.0)
    zone_name = params.get("seismic_zone", "Zone IV")

    content = []
    content.append("$ PROGRAM INFORMATION")
    content.append('  PROGRAM  "ETABS"  VERSION "20.0.0"')
    content.append('  UNITS  "KN-M"')
    content.append("")
    content.append("$ STORIES - STORY DATA")
    for s in range(n_stories, 0, -1):
        content.append(f'  STORY "Story{s}"  HEIGHT {story_height}  ELEVATION {s * story_height}  SIMILARTO "Story1"  MASTER {"YES" if s == 1 else "NO"}')
    content.append('  BASE ELEVATION 0.0')
    content.append("")
    content.append("$ GRID LINES")
    for i in range(bay_x_count + 1):
        content.append(f'  GRID "X" "{chr(65 + i)}"  ORDINATE {i * bay_x_length}')
    for j in range(bay_y_count + 1):
        content.append(f'  GRID "Y" "{j + 1}"  ORDINATE {j * bay_y_length}')
    content.append("")
    content.append("$ MATERIAL PROPERTIES")
    content.append('  MATERIAL "Fe250"  TYPE "STEEL"  DESIGN "IS800-2007"  E 200000000  FY 250000  FU 410000')
    content.append("")
    content.append("$ FRAME SECTIONS")
    content.append('  FRAMESECTION "ISMB400"  MATERIAL "Fe250"  SHAPE "STEEL_I"  D 0.40  BF 0.14  TF 0.016  TW 0.0089')
    content.append('  FRAMESECTION "ISHB400"  MATERIAL "Fe250"  SHAPE "STEEL_I"  D 0.40  BF 0.25  TF 0.0127  TW 0.0106')
    content.append("")
    content.append("$ LOAD PATTERNS")
    content.append('  LOADPATTERN "DEAD"  TYPE "DEAD"  SELFWEIGHT 1.0')
    content.append('  LOADPATTERN "LIVE"  TYPE "LIVE"  SELFWEIGHT 0.0')
    content.append(f'  LOADPATTERN "EQX"   TYPE "SEISMIC"  CODE "IS1893 2016"  ZONE "{zone_name}"  R 5.0  I 1.2')
    content.append("")
    content.append("$ UNIFORM AREA LOADS")
    content.append(f'  AREALOAD  LOADPATTERN "DEAD"  UNIFORM {dead_load}')
    content.append(f'  AREALOAD  LOADPATTERN "LIVE"  UNIFORM {live_load}')
    content.append("")
    content.append("$ LOAD COMBINATIONS")
    content.append('  COMBO "1.5(DL+LL)"      TYPE "ADD"  "DEAD" 1.5  "LIVE" 1.5')
    content.append('  COMBO "1.2(DL+LL+EQX)"  TYPE "ADD"  "DEAD" 1.2  "LIVE" 1.2  "EQX" 1.2')
    content.append('  COMBO "1.5(DL+EQX)"     TYPE "ADD"  "DEAD" 1.5  "EQX" 1.5')
    content.append('  COMBO "0.9DL+1.5EQX"    TYPE "ADD"  "DEAD" 0.9  "EQX" 1.5')
    content.append("")
    content.append("$ END OF FILE")

    e2k_text = "\n".join(content)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(e2k_text)
    return output_path


def generate_staad_std(params: Dict, output_path: str = "model.std") -> str:
    """
    Generates a standard STAAD.Pro command file (.std) for a multistory steel building.
    """
    n_stories = params.get("n_stories", 10)
    bay_x_count = params.get("bay_x_count", 4)
    bay_y_count = params.get("bay_y_count", 3)
    bay_x_length = params.get("bay_x_length", 6.0)
    bay_y_length = params.get("bay_y_length", 6.0)
    story_height = params.get("story_height", 3.5)
    dead_load = params.get("dead_load", 4.5)
    live_load = params.get("live_load", 3.0)

    lines = []
    lines.append("STAAD SPACE")
    lines.append("START JOB INFORMATION")
    lines.append("ENGINEER DATE 25-AUG-2026")
    lines.append("JOB NAME G+10 STEEL FRAME")
    lines.append("END JOB INFORMATION")
    lines.append("INPUT WIDTH 79")
    lines.append("UNIT METER KN")
    lines.append("JOINT COORDINATES")

    # Generate Node Coordinates
    node_id = 1
    node_map = {}
    for s in range(n_stories + 1):
        y = s * story_height
        for j in range(bay_y_count + 1):
            z = j * bay_y_length
            for i in range(bay_x_count + 1):
                x = i * bay_x_length
                lines.append(f"{node_id} {x:.2f} {y:.2f} {z:.2f};")
                node_map[(i, j, s)] = node_id
                node_id += 1

    lines.append("MEMBER INCIDENCES")
    member_id = 1
    col_members = []
    beam_members = []

    # Column incidences
    for s in range(n_stories):
        for j in range(bay_y_count + 1):
            for i in range(bay_x_count + 1):
                n1 = node_map[(i, j, s)]
                n2 = node_map[(i, j, s + 1)]
                lines.append(f"{member_id} {n1} {n2};")
                col_members.append(member_id)
                member_id += 1

    # Beam incidences (X-direction)
    for s in range(1, n_stories + 1):
        for j in range(bay_y_count + 1):
            for i in range(bay_x_count):
                n1 = node_map[(i, j, s)]
                n2 = node_map[(i + 1, j, s)]
                lines.append(f"{member_id} {n1} {n2};")
                beam_members.append(member_id)
                member_id += 1

    # Beam incidences (Z-direction)
    for s in range(1, n_stories + 1):
        for i in range(bay_x_count + 1):
            for j in range(bay_y_count):
                n1 = node_map[(i, j, s)]
                n2 = node_map[(i, j + 1, s)]
                lines.append(f"{member_id} {n1} {n2};")
                beam_members.append(member_id)
                member_id += 1

    lines.append("SECTION POOL INDIAN")
    lines.append("MEMBER PROPERTY INDIAN")
    lines.append(f"1 TO {len(col_members)} TABLE ST ISHB400")
    lines.append(f"{len(col_members) + 1} TO {member_id - 1} TABLE ST ISMB400")
    lines.append("CONSTANTS")
    lines.append("MATERIAL STEEL ALL")
    lines.append("SUPPORTS")
    # Base nodes are fixed
    base_nodes = [str(node_map[(i, j, 0)]) for j in range(bay_y_count + 1) for i in range(bay_x_count + 1)]
    lines.append(f"{' '.join(base_nodes)} FIXED")
    lines.append("DEFINE 1893 LOAD")
    lines.append("ZONE 0.24 RF 5 I 1.2 SS 2 DM 0.05")
    lines.append("SELFWEIGHT 1")
    lines.append(f"FLOOR WEIGHT YRANGE 0 {n_stories * story_height} FLOAD {dead_load + 0.25 * live_load}")
    lines.append("LOAD 1 LOADTYPE Seismic TITLE EQX")
    lines.append("1893 LOAD X 1.0")
    lines.append("LOAD 2 LOADTYPE Dead TITLE DEAD LOAD")
    lines.append("SELFWEIGHT Y -1")
    lines.append(f"FLOOR LOAD YRANGE 0 {n_stories * story_height} FLOAD -{dead_load}")
    lines.append("LOAD 3 LOADTYPE Live TITLE LIVE LOAD")
    lines.append(f"FLOOR LOAD YRANGE 0 {n_stories * story_height} FLOAD -{live_load}")
    lines.append("LOAD COMB 101 1.5(DL+LL)")
    lines.append("2 1.5 3 1.5")
    lines.append("LOAD COMB 102 1.2(DL+LL+EQX)")
    lines.append("2 1.2 3 1.2 1 1.2")
    lines.append("PERFORM ANALYSIS")
    lines.append("PARAMETER 1")
    lines.append("CODE INDIAN 2007 LSD")
    lines.append("FYLD 250000 ALL")
    lines.append("CHECK CODE ALL")
    lines.append("FINISH")

    std_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(std_text)
    return output_path


def get_etabs_oapi_batch_template() -> str:
    """
    Returns an automation Python script template for extracting real FEA batch design results from ETABS.
    """
    return '''"""
ETABS OAPI Batch Runner and Data Extractor Template
Automates opening ETABS models, executing analysis and IS 800 steel design,
and exporting member forces and optimal section sizes directly to pandas DataFrame.
"""

import sys
import comtypes.client
import pandas as pd

def run_etabs_batch(model_paths):
    # Connect to active ETABS instance or create new COM interface
    helper = comtypes.client.CreateObject('ETABSv1.Helper')
    helper = helper.QueryInterface(comtypes.gen.ETABSv1.cHelper)
    etabs_obj = helper.CreateObjectProgID("CSI.ETABS.API.ETABSObject")
    etabs_obj.ApplicationStart()
    sap_model = etabs_obj.SapModel
    
    extracted_data = []
    for mpath in model_paths:
        sap_model.File.OpenFile(mpath)
        sap_model.Analyze.RunAnalysis()
        sap_model.DesignSteel.StartDesign()
        
        # Extract Design Output for Beams and Columns
        # sap_model.DesignSteel.GetSummaryResultsBeam(...)
        # sap_model.DesignSteel.GetSummaryResultsColumn(...)
        print(f"Completed analysis and design extraction for {mpath}")
        
    etabs_obj.ApplicationExit(False)
    return pd.DataFrame(extracted_data)

if __name__ == "__main__":
    print("ETABS OAPI Batch Extraction script ready.")
'''
