"""
Self-contained preprocessing for the multimodal sepsis TFM pipeline.

This script replaces the dependency on the existing `src/` preprocessing code.
It extracts the task-specific tables from raw MIMIC-IV CSV files and builds
`onset.csv` (a legacy artifact: Stage 2 recomputes the label into `sepsis3_onset.csv`,
which is what defines the early sepsis prediction task).

Pipeline position: Stage 1 (first script to run)
Inputs:  Raw MIMIC-IV v3.x CSV files under --mimic_dir
Outputs: data/01_extracted/ (raw event tables)
         data/02_onset/     (onset.csv + processed helper files)
"""

import argparse
import json
import os
import time

import duckdb
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# MIMIC-IV itemid constants
# Each itemid is a numeric code in MIMIC's d_items table that identifies
# a specific measurement type recorded in chartevents or inputevents.
# ---------------------------------------------------------------------------

# Chartevents itemids that represent bedside culture-specimen collection
# (blood, urine, respiratory, wound cultures, etc.).  Used to detect
# suspected infection alongside antibiotic administration.
CULTURE_ITEMIDS = (
    225401, 225437, 225444, 225451, 225454, 225814, 225816, 225817, 225818,
    225722, 225723, 225724, 225725, 225726, 225727, 225728, 225729, 225730,
    225731, 225732, 225733, 227726, 225734, 225735, 225736, 225768, 226131,
)

# Chartevents itemids related to mechanical ventilation settings and status
# (ventilator mode, tidal volume, PEEP, FiO2 delivery device, etc.).
# Used to produce the binary `mechvent` flag per charttime.
MECHVENT_ITEMIDS = (
    223894, 226732, 224687, 224685, 224684, 224686, 224697, 224695, 224696,
    224746, 224747, 226873, 224738, 224419, 224750, 227187, 224707, 224709,
    224705, 224706, 220339, 224700, 224702, 227809, 227810, 224701,
)

# Glasgow Coma Scale component itemids (MIMIC-IV chartevents).
# Stored with numeric valuenum: eye 1-4, verbal 1-5, motor 1-6.
# The intubated verbal case "No Response-ETT" has valuenum = 1, so summed
# GCS stays within the clinically valid [3, 15] range.
GCS_ITEMIDS = (220739, 223900, 223901)

# Vasopressor infusion itemids (inputevents): norepinephrine (221906, 221289),
# vasopressin (222315), phenylephrine (221749), dopamine (221662).
# Rates from different drugs are normalised to a norepinephrine-equivalent
# dose in extract_inputevents_all.
VASO_ITEMIDS = (221749, 221906, 221289, 222315, 221662)

# IV fluid infusion itemids (inputevents): normal saline, lactated Ringer's,
# albumin, packed red blood cells, fresh frozen plasma, and similar.
# The `tev` (tonicity-equivalent volume) column rescales each fluid type to
# an effective volume contribution using the CASE multipliers below.
FLUID_ITEMIDS = (
    225158, 225943, 226089, 225168, 225828, 220862, 220970, 220864, 225159,
    220995, 225170, 225825, 227533, 225161, 227531, 225171, 225827, 225941,
    225823, 228341,
)

# Pre-admission fluid items recorded in inputevents; summed to a single
# `inputpreadm` scalar per stay and exported to preadm_fluid.csv.
PREADM_FLUID_ITEMIDS = (
    226361, 226363, 226364, 226365, 226367, 226368, 226369, 226370, 226371,
    226372, 226375, 226376, 227070, 227071, 227072,
)

# Urine output itemids (outputevents): Foley catheter, nephrostomy, and
# other drainage routes.  Values are summed over the SOFA trailing window
# to score the renal component.
UO_ITEMIDS = (
    226559, 226560, 227510, 226561, 227489, 226584, 226563, 226564, 226565,
    226557, 226558, 226713, 226567,
)

# Generic Sequence Numbers (GSN) from the hospital formulary that map to
# systemic antibiotic prescriptions in MIMIC-IV's prescriptions table.
# GSN is a drug classification code similar to NDC but provider-specific.
# This list is taken from the MIT-LCP sepsis3.sql antibiotic GSN list and
# covers broad-spectrum and targeted antibiotics used in ICU practice.
ANTIBIOTIC_GSN_CODES = (
    "002542", "002543", "007371", "008873", "008877", "008879", "008880",
    "008935", "008941", "008942", "008943", "008944", "008983", "008984",
    "008990", "008991", "008992", "008995", "008996", "008998", "009043",
    "009046", "009065", "009066", "009136", "009137", "009162", "009164",
    "009165", "009171", "009182", "009189", "009213", "009214", "009218",
    "009219", "009221", "009226", "009227", "009235", "009242", "009263",
    "009273", "009284", "009298", "009299", "009310", "009322", "009323",
    "009326", "009327", "009339", "009346", "009351", "009354", "009362",
    "009394", "009395", "009396", "009509", "009510", "009511", "009544",
    "009585", "009591", "009592", "009630", "013023", "013645", "013723",
    "013724", "013725", "014182", "014500", "015979", "016368", "016373",
    "016408", "016931", "016932", "016949", "018636", "018637", "018766",
    "019283", "021187", "021205", "021735", "021871", "023372", "023989",
    "024095", "024194", "024668", "025080", "026721", "027252", "027465",
    "027470", "029325", "029927", "029928", "037042", "039551", "039806",
    "040819", "041798", "043350", "043879", "044143", "045131", "045132",
    "046771", "047797", "048077", "048262", "048266", "048292", "049835",
    "050442", "050443", "051932", "052050", "060365", "066295", "067471",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MIMIC-IV tables and build onset labels.")
    parser.add_argument("--mimic_dir", default="physionet.org/files/mimiciv/3.1")
    parser.add_argument("--extracted_dir", default="data/01_extracted", help="Output directory for raw extraction tables.")
    parser.add_argument("--onset_dir", default="data/02_onset", help="Output directory for onset and processed helper files.")
    parser.add_argument("--mapping_file", default="multi_modality_code/reference_files/measurement_mappings.json")
    parser.add_argument("--separator", default="|")
    parser.add_argument(
        "--lead_time",
        type=int,
        default=6,
        help="LEGACY: affects only onset.csv, never the final label. The lead time that "
        "defines the task is the one passed to stage 2 (3_build_sepsis3_labels.py).",
    )
    parser.add_argument("--control_ratio", type=float, default=1.0, help="Control-to-positive ratio, capped at 1.")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip DuckDB extraction and run onset only.")
    parser.add_argument("--skip_onset", action="store_true", help="Run extraction only, without onset generation.")
    parser.add_argument(
        "--only_demog",
        action="store_true",
        help="Re-extract demographics only (demog.csv + demog_processed.csv) and exit. Nothing "
        "else in Stage 1 depends on the demographics query, so this refreshes the static "
        "features -- Charlson, diagnosis_count -- in minutes instead of re-reading chartevents, "
        "labevents and prescriptions. The downstream label artifact is untouched: Stage 2 reads "
        "demographics only for the eligibility filters and the intime/outtime bounds, and this "
        "query changes neither.",
    )
    return parser.parse_args()


def timed(name):
    """Decorator that prints the step name and wall-clock duration on completion."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"\n{name}")
            print("-" * len(name))
            start = time.time()
            result = func(*args, **kwargs)
            print(f"Done in {time.time() - start:.1f}s")
            return result
        return wrapper
    return decorator


class MIMICPaths:
    """Resolves full paths to MIMIC-IV CSV files under the hosp/ and icu/ subdirs."""

    def __init__(self, mimic_dir: str):
        self.hosp = os.path.join(mimic_dir, "hosp")
        self.icu = os.path.join(mimic_dir, "icu")

    def h(self, name: str) -> str:
        """Return full path for a hospital-module file."""
        return os.path.join(self.hosp, name)

    def i(self, name: str) -> str:
        """Return full path for an ICU-module file."""
        return os.path.join(self.icu, name)


def csv_list(values) -> str:
    """Convert an iterable to a comma-separated string for SQL IN clauses."""
    return ",".join(str(value) for value in values)


def sql_string_list(values) -> str:
    """Convert an iterable to a quoted comma-separated string for SQL IN clauses."""
    return ",".join(f"'{value}'" for value in values)


def load_mapping_codes(mapping_file: str) -> tuple[set[int], set[int], set[int]]:
    """
    Parse measurement_mappings.json and return three itemid sets:
    - all_codes: every itemid in the mapping (vitals + labs)
    - vital_codes: items categorised as vital/respiratory/neurological/hemodynamic
    - lab_codes: items categorised as laboratory or blood_gas
    """
    with open(mapping_file) as file:
        mapping = json.load(file)

    vital_codes = set()
    lab_codes = set()
    all_codes = set()
    for info in mapping.values():
        codes = {int(code) for code in info["codes"]}
        all_codes.update(codes)
        if info.get("category") in {"laboratory", "blood_gas"}:
            lab_codes.update(codes)
        else:
            vital_codes.update(codes)
    return all_codes, vital_codes, lab_codes


def ensure_inputs(paths: MIMICPaths) -> None:
    """Raise FileNotFoundError early if any required MIMIC-IV source file is absent."""
    required = [
        paths.h("admissions.csv"),
        paths.h("patients.csv"),
        paths.h("diagnoses_icd.csv"),
        paths.h("prescriptions.csv"),
        paths.h("microbiologyevents.csv"),
        paths.h("labevents.csv"),
        paths.i("icustays.csv"),
        paths.i("chartevents.csv"),
        paths.i("inputevents.csv"),
        paths.i("outputevents.csv"),
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing MIMIC-IV files:\n" + "\n".join(missing))


@timed("Extract ICU stays")
def extract_icustays(con, paths: MIMICPaths, output_dir: str, sep: str) -> None:
    """Copy icustays.csv verbatim, sorted by subject/admission/stay."""
    con.execute(f"""
        COPY (
            SELECT *
            FROM read_csv_auto('{paths.i("icustays.csv")}')
            ORDER BY subject_id, hadm_id, stay_id
        ) TO '{output_dir}/icustays.csv' (DELIMITER '{sep}', HEADER);
    """)


# ---------------------------------------------------------------------------
# Charlson comorbidity index
# ---------------------------------------------------------------------------
# The 17 category definitions and the weights below are transcribed from the
# official MIMIC-IV concept `mimic-iv/concepts_duckdb/comorbidity/charlson.sql`
# (vendored in this repo under `stuff/mimic-code/`), which implements the Quan
# et al. (2005) enhanced ICD-9-CM / ICD-10 coding maps. Keeping the definition
# byte-identical is the point: it makes `charlson_comorbidity_index` comparable
# with any other MIMIC study that uses the derived tables.
#
# Two deliberate departures from that script, neither of which changes a value:
# the flags are grouped from `diagnoses_icd` alone and COALESCEd to 0 (the
# official version groups over a second read of `admissions` to materialise the
# all-zero rows), and `age_score` reuses the age this script already derives
# from anchor_age/anchor_year instead of `mimiciv_derived.age`, which is the
# same expression.
#
# Interpretation caveat, to be carried into the memoir: ICD codes in MIMIC are
# assigned for billing at *discharge*, so no comorbidity flag is strictly
# "known at prediction time". Charlson is still the more defensible of the two
# comorbidity features here, because it only counts chronic conditions --
# `diagnosis_count` is the whole admission's coding burden, acute codes for the
# very deterioration being predicted included.
CHARLSON_CATEGORIES_SQL = """
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 3) IN ('410', '412')
                    OR SUBSTR(icd10_code, 1, 3) IN ('I21', 'I22')
                    OR SUBSTR(icd10_code, 1, 4) = 'I252'
                THEN 1 ELSE 0 END) AS myocardial_infarct,
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 3) = '428'
                    OR SUBSTR(icd9_code, 1, 5) IN ('39891', '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493')
                    OR SUBSTR(icd9_code, 1, 4) BETWEEN '4254' AND '4259'
                    OR SUBSTR(icd10_code, 1, 3) IN ('I43', 'I50')
                    OR SUBSTR(icd10_code, 1, 4) IN ('I099', 'I110', 'I130', 'I132', 'I255', 'I420', 'I425', 'I426', 'I427', 'I428', 'I429', 'P290')
                THEN 1 ELSE 0 END) AS congestive_heart_failure,
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 3) IN ('440', '441')
                    OR SUBSTR(icd9_code, 1, 4) IN ('0930', '4373', '4471', '5571', '5579', 'V434')
                    OR SUBSTR(icd9_code, 1, 4) BETWEEN '4431' AND '4439'
                    OR SUBSTR(icd10_code, 1, 3) IN ('I70', 'I71')
                    OR SUBSTR(icd10_code, 1, 4) IN ('I731', 'I738', 'I739', 'I771', 'I790', 'I792', 'K551', 'K558', 'K559', 'Z958', 'Z959')
                THEN 1 ELSE 0 END) AS peripheral_vascular_disease,
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 3) BETWEEN '430' AND '438'
                    OR SUBSTR(icd9_code, 1, 5) = '36234'
                    OR SUBSTR(icd10_code, 1, 3) IN ('G45', 'G46')
                    OR SUBSTR(icd10_code, 1, 3) BETWEEN 'I60' AND 'I69'
                    OR SUBSTR(icd10_code, 1, 4) = 'H340'
                THEN 1 ELSE 0 END) AS cerebrovascular_disease,
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 3) = '290'
                    OR SUBSTR(icd9_code, 1, 4) IN ('2941', '3312')
                    OR SUBSTR(icd10_code, 1, 3) IN ('F00', 'F01', 'F02', 'F03', 'G30')
                    OR SUBSTR(icd10_code, 1, 4) IN ('F051', 'G311')
                THEN 1 ELSE 0 END) AS dementia,
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 3) BETWEEN '490' AND '505'
                    OR SUBSTR(icd9_code, 1, 4) IN ('4168', '4169', '5064', '5081', '5088')
                    OR SUBSTR(icd10_code, 1, 3) BETWEEN 'J40' AND 'J47'
                    OR SUBSTR(icd10_code, 1, 3) BETWEEN 'J60' AND 'J67'
                    OR SUBSTR(icd10_code, 1, 4) IN ('I278', 'I279', 'J684', 'J701', 'J703')
                THEN 1 ELSE 0 END) AS chronic_pulmonary_disease,
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 3) = '725'
                    OR SUBSTR(icd9_code, 1, 4) IN ('4465', '7100', '7101', '7102', '7103', '7104', '7140', '7141', '7142', '7148')
                    OR SUBSTR(icd10_code, 1, 3) IN ('M05', 'M06', 'M32', 'M33', 'M34')
                    OR SUBSTR(icd10_code, 1, 4) IN ('M315', 'M351', 'M353', 'M360')
                THEN 1 ELSE 0 END) AS rheumatic_disease,
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 3) IN ('531', '532', '533', '534')
                    OR SUBSTR(icd10_code, 1, 3) IN ('K25', 'K26', 'K27', 'K28')
                THEN 1 ELSE 0 END) AS peptic_ulcer_disease,
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 3) IN ('570', '571')
                    OR SUBSTR(icd9_code, 1, 4) IN ('0706', '0709', '5733', '5734', '5738', '5739', 'V427')
                    OR SUBSTR(icd9_code, 1, 5) IN ('07022', '07023', '07032', '07033', '07044', '07054')
                    OR SUBSTR(icd10_code, 1, 3) IN ('B18', 'K73', 'K74')
                    OR SUBSTR(icd10_code, 1, 4) IN ('K700', 'K701', 'K702', 'K703', 'K709', 'K713', 'K714', 'K715', 'K717', 'K760', 'K762', 'K763', 'K764', 'K768', 'K769', 'Z944')
                THEN 1 ELSE 0 END) AS mild_liver_disease,
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 4) IN ('2500', '2501', '2502', '2503', '2508', '2509')
                    OR SUBSTR(icd10_code, 1, 4) IN ('E100', 'E101', 'E106', 'E108', 'E109', 'E110', 'E111', 'E116', 'E118', 'E119', 'E120', 'E121', 'E126', 'E128', 'E129', 'E130', 'E131', 'E136', 'E138', 'E139', 'E140', 'E141', 'E146', 'E148', 'E149')
                THEN 1 ELSE 0 END) AS diabetes_without_cc,
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 4) IN ('2504', '2505', '2506', '2507')
                    OR SUBSTR(icd10_code, 1, 4) IN ('E102', 'E103', 'E104', 'E105', 'E107', 'E112', 'E113', 'E114', 'E115', 'E117', 'E122', 'E123', 'E124', 'E125', 'E127', 'E132', 'E133', 'E134', 'E135', 'E137', 'E142', 'E143', 'E144', 'E145', 'E147')
                THEN 1 ELSE 0 END) AS diabetes_with_cc,
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 3) IN ('342', '343')
                    OR SUBSTR(icd9_code, 1, 4) IN ('3341', '3440', '3441', '3442', '3443', '3444', '3445', '3446', '3449')
                    OR SUBSTR(icd10_code, 1, 3) IN ('G81', 'G82')
                    OR SUBSTR(icd10_code, 1, 4) IN ('G041', 'G114', 'G801', 'G802', 'G830', 'G831', 'G832', 'G833', 'G834', 'G839')
                THEN 1 ELSE 0 END) AS paraplegia,
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 3) IN ('582', '585', '586', 'V56')
                    OR SUBSTR(icd9_code, 1, 4) IN ('5880', 'V420', 'V451')
                    OR SUBSTR(icd9_code, 1, 4) BETWEEN '5830' AND '5837'
                    OR SUBSTR(icd9_code, 1, 5) IN ('40301', '40311', '40391', '40402', '40403', '40412', '40413', '40492', '40493')
                    OR SUBSTR(icd10_code, 1, 3) IN ('N18', 'N19')
                    OR SUBSTR(icd10_code, 1, 4) IN ('I120', 'I131', 'N032', 'N033', 'N034', 'N035', 'N036', 'N037', 'N052', 'N053', 'N054', 'N055', 'N056', 'N057', 'N250', 'Z490', 'Z491', 'Z492', 'Z940', 'Z992')
                THEN 1 ELSE 0 END) AS renal_disease,
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 3) BETWEEN '140' AND '172'
                    OR SUBSTR(icd9_code, 1, 4) BETWEEN '1740' AND '1958'
                    OR SUBSTR(icd9_code, 1, 3) BETWEEN '200' AND '208'
                    OR SUBSTR(icd9_code, 1, 4) = '2386'
                    OR SUBSTR(icd10_code, 1, 3) IN ('C43', 'C88')
                    OR SUBSTR(icd10_code, 1, 3) BETWEEN 'C00' AND 'C26'
                    OR SUBSTR(icd10_code, 1, 3) BETWEEN 'C30' AND 'C34'
                    OR SUBSTR(icd10_code, 1, 3) BETWEEN 'C37' AND 'C41'
                    OR SUBSTR(icd10_code, 1, 3) BETWEEN 'C45' AND 'C58'
                    OR SUBSTR(icd10_code, 1, 3) BETWEEN 'C60' AND 'C76'
                    OR SUBSTR(icd10_code, 1, 3) BETWEEN 'C81' AND 'C85'
                    OR SUBSTR(icd10_code, 1, 3) BETWEEN 'C90' AND 'C97'
                THEN 1 ELSE 0 END) AS malignant_cancer,
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 4) IN ('4560', '4561', '4562')
                    OR SUBSTR(icd9_code, 1, 4) BETWEEN '5722' AND '5728'
                    OR SUBSTR(icd10_code, 1, 4) IN ('I850', 'I859', 'I864', 'I982', 'K704', 'K711', 'K721', 'K729', 'K765', 'K766', 'K767')
                THEN 1 ELSE 0 END) AS severe_liver_disease,
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 3) IN ('196', '197', '198', '199')
                    OR SUBSTR(icd10_code, 1, 3) IN ('C77', 'C78', 'C79', 'C80')
                THEN 1 ELSE 0 END) AS metastatic_solid_tumor,
                MAX(CASE WHEN
                    SUBSTR(icd9_code, 1, 3) IN ('042', '043', '044')
                    OR SUBSTR(icd10_code, 1, 3) IN ('B20', 'B21', 'B22', 'B24')
                THEN 1 ELSE 0 END) AS aids
"""

# The 17 flags in the order they are emitted, so downstream code can list them
# without re-parsing the SQL.
CHARLSON_FLAG_COLUMNS = (
    "myocardial_infarct",
    "congestive_heart_failure",
    "peripheral_vascular_disease",
    "cerebrovascular_disease",
    "dementia",
    "chronic_pulmonary_disease",
    "rheumatic_disease",
    "peptic_ulcer_disease",
    "mild_liver_disease",
    "diabetes_without_cc",
    "diabetes_with_cc",
    "paraplegia",
    "renal_disease",
    "malignant_cancer",
    "severe_liver_disease",
    "metastatic_solid_tumor",
    "aids",
)

# Charlson weights: 1 point each for the mild categories, and the three
# GREATEST() pairs implement the "worse form supersedes the milder one" rule
# (severe liver disease 3 replaces mild liver disease 1; diabetes with chronic
# complications 2 replaces diabetes without 1; metastatic solid tumour 6
# replaces localised malignancy 2).
CHARLSON_INDEX_SQL = """
                age_score
                + myocardial_infarct
                + congestive_heart_failure
                + peripheral_vascular_disease
                + cerebrovascular_disease
                + dementia
                + chronic_pulmonary_disease
                + rheumatic_disease
                + peptic_ulcer_disease
                + GREATEST(mild_liver_disease, 3 * severe_liver_disease)
                + GREATEST(2 * diabetes_with_cc, diabetes_without_cc)
                + GREATEST(2 * malignant_cancer, 6 * metastatic_solid_tumor)
                + 2 * paraplegia
                + 2 * renal_disease
                + 6 * aids
"""


@timed("Extract demographics")
def extract_demog(con, paths: MIMICPaths, output_dir: str, sep: str) -> None:
    """
    Join admissions + icustays + patients + diagnosis counts + Charlson
    comorbidities into one row per ICU stay.

    Key derived columns:
    - adm_order: rank of this stay within the subject (1 = first ICU admission)
    - unit: encoded ICU care unit type (1=MICU, 2=SICU, 3=TSICU, 4=CSRU, 5=NeuroInt, 6=CCU, 0=other)
    - age: calculated from MIMIC's anchor_age/anchor_year mechanism
    - diagnosis_count: number of ICD rows for this admission (crude coding-burden proxy)
    - the 17 Charlson category flags, `age_score`, and the weighted
      `charlson_comorbidity_index` (see CHARLSON_CATEGORIES_SQL above)

    All timestamps are stored as Unix epoch seconds for consistent arithmetic.
    """
    con.execute(f"""
        COPY (
            WITH diagnosis_counts AS (
                SELECT hadm_id, COUNT(*) AS diagnosis_count
                FROM read_csv_auto('{paths.h("diagnoses_icd.csv")}')
                GROUP BY hadm_id
            ),
            diag AS (
                SELECT
                    hadm_id,
                    CASE WHEN icd_version = 9 THEN icd_code END AS icd9_code,
                    CASE WHEN icd_version = 10 THEN icd_code END AS icd10_code
                FROM read_csv_auto('{paths.h("diagnoses_icd.csv")}')
            ),
            comorbidities AS (
                SELECT
                    hadm_id,
{CHARLSON_CATEGORIES_SQL.rstrip()}
                FROM diag
                GROUP BY hadm_id
            ),
            base AS (
                SELECT
                    ad.subject_id,
                    ad.hadm_id,
                    i.stay_id,
                    epoch(ad.admittime::TIMESTAMP) AS admittime,
                    epoch(ad.dischtime::TIMESTAMP) AS dischtime,
                    ROW_NUMBER() OVER (PARTITION BY ad.subject_id ORDER BY i.intime ASC) AS adm_order,
                    CASE
                        WHEN i.first_careunit = 'Neuro Intermediate' THEN 5
                        WHEN i.first_careunit LIKE '%SICU%' THEN 2
                        WHEN i.first_careunit LIKE '%CSRU%' OR i.first_careunit LIKE '%Cardiac%' THEN 4
                        WHEN i.first_careunit LIKE '%CCU%' OR i.first_careunit LIKE '%Coronary%' THEN 6
                        WHEN i.first_careunit LIKE '%MICU%' OR i.first_careunit LIKE '%Medical%' THEN 1
                        WHEN i.first_careunit LIKE '%TSICU%' OR i.first_careunit LIKE '%Trauma%' THEN 3
                        ELSE 0
                    END AS unit,
                    epoch(i.intime::TIMESTAMP) AS intime,
                    epoch(i.outtime::TIMESTAMP) AS outtime,
                    i.los,
                    pa.anchor_age + EXTRACT(YEAR FROM ad.admittime::TIMESTAMP) - pa.anchor_year AS age,
                    CASE WHEN pa.gender = 'M' THEN 1 WHEN pa.gender = 'F' THEN 2 END AS gender,
                    COALESCE(dc.diagnosis_count, 0) AS diagnosis_count
                FROM read_csv_auto('{paths.h("admissions.csv")}') ad
                INNER JOIN read_csv_auto('{paths.i("icustays.csv")}') i ON ad.hadm_id = i.hadm_id
                INNER JOIN read_csv_auto('{paths.h("patients.csv")}') pa ON pa.subject_id = i.subject_id
                LEFT JOIN diagnosis_counts dc ON ad.hadm_id = dc.hadm_id
            ),
            flagged AS (
                SELECT
                    base.*,
                    CASE
                        WHEN base.age <= 50 THEN 0
                        WHEN base.age <= 60 THEN 1
                        WHEN base.age <= 70 THEN 2
                        WHEN base.age <= 80 THEN 3
                        ELSE 4
                    END AS age_score,
                    {", ".join(f"COALESCE(co.{flag}, 0) AS {flag}" for flag in CHARLSON_FLAG_COLUMNS)}
                FROM base
                LEFT JOIN comorbidities co ON base.hadm_id = co.hadm_id
            )
            SELECT
                *,
{CHARLSON_INDEX_SQL.rstrip()} AS charlson_comorbidity_index
            FROM flagged
            ORDER BY subject_id ASC, intime ASC
        ) TO '{output_dir}/demog.csv' (DELIMITER '{sep}', HEADER);
    """)


@timed("Extract charted events, labs, culture events, ventilation, and GCS")
def extract_chartevents_all(con, paths: MIMICPaths, output_dir: str, sep: str, mapping_file: str) -> None:
    """
    Single pass over chartevents.csv to produce five output files:
    - culture.csv     : culture-collection events (suspected infection detector)
    - chartevents.csv : vital signs and respiratory measurements (numeric)
    - labs_ce.csv     : lab-type measurements charted at bedside (e.g. iStat)
    - mechvent.csv    : binary mechanical-ventilation flag per (stay_id, charttime)
    - gcs.csv         : summed GCS score (eye + verbal + motor)

    A temporary DuckDB table `ce_filtered` is created for the full itemid scan
    and dropped at the end to avoid re-reading the ~30 GB chartevents file.

    The CASE block for vital itemids normalises oxygen delivery device codes
    (itemid 223834/226732) from free-text strings to ordinal integers and
    maps RASS sedation scores (228096) from prefixed text to signed integers.
    """
    all_mapping_codes, vital_codes, lab_codes = load_mapping_codes(mapping_file)
    all_itemids = sorted(
        all_mapping_codes | set(CULTURE_ITEMIDS) | set(MECHVENT_ITEMIDS) | set(GCS_ITEMIDS)
    )

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ce_filtered AS
        SELECT
            subject_id,
            hadm_id,
            stay_id,
            epoch(charttime::TIMESTAMP) AS charttime,
            itemid,
            value,
            valuenum
        FROM read_csv_auto('{paths.i("chartevents.csv")}')
        WHERE itemid IN ({csv_list(all_itemids)})
    """)

    con.execute(f"""
        COPY (
            SELECT subject_id, hadm_id, stay_id, charttime, itemid
            FROM ce_filtered
            WHERE itemid IN ({csv_list(CULTURE_ITEMIDS)})
            ORDER BY subject_id, hadm_id, charttime
        ) TO '{output_dir}/culture.csv' (DELIMITER '{sep}', HEADER);
    """)

    con.execute(f"""
        COPY (
            SELECT DISTINCT stay_id, charttime, itemid,
                CASE
                    WHEN itemid IN (223834, 226732) AND value = 'None' THEN 0
                    WHEN itemid IN (223834, 226732) AND value = 'Nasal cannula' THEN 2
                    WHEN itemid IN (223834, 226732) AND value = 'Face tent' THEN 3
                    WHEN itemid IN (223834, 226732) AND value = 'Aerosol-cool' THEN 4
                    WHEN itemid IN (223834, 226732) AND value = 'Trach mask' THEN 5
                    WHEN itemid IN (223834, 226732) AND value LIKE 'High flow%' THEN 6
                    WHEN itemid IN (223834, 226732) AND value = 'Non-rebreather' THEN 7
                    WHEN itemid IN (223834, 226732) AND value = 'Venti mask' THEN 8
                    WHEN itemid IN (223834, 226732) AND value = 'Endotracheal tube' THEN 10
                    WHEN itemid IN (223834, 226732) AND value = 'Tracheostomy tube' THEN 11
                    WHEN itemid = 228096 AND value LIKE '0%' THEN 0
                    WHEN itemid = 228096 AND value LIKE '-1%' THEN -1
                    WHEN itemid = 228096 AND value LIKE '-2%' THEN -2
                    WHEN itemid = 228096 AND value LIKE '-3%' THEN -3
                    WHEN itemid = 228096 AND value LIKE '-4%' THEN -4
                    WHEN itemid = 228096 AND value LIKE '-5%' THEN -5
                    WHEN itemid = 228096 AND value LIKE '+1%' THEN 1
                    WHEN itemid = 228096 AND value LIKE '+2%' THEN 2
                    WHEN itemid = 228096 AND value LIKE '+3%' THEN 3
                    WHEN itemid = 228096 AND value LIKE '+4%' THEN 4
                    ELSE valuenum
                END AS valuenum
            FROM ce_filtered
            WHERE itemid IN ({csv_list(vital_codes)})
              AND stay_id IS NOT NULL
              AND (value IS NOT NULL OR valuenum IS NOT NULL)
            ORDER BY stay_id, charttime
        ) TO '{output_dir}/chartevents.csv' (DELIMITER '{sep}', HEADER);
    """)

    con.execute(f"""
        COPY (
            SELECT stay_id, charttime, itemid, valuenum
            FROM ce_filtered
            WHERE itemid IN ({csv_list(lab_codes)})
              AND stay_id IS NOT NULL
              AND valuenum IS NOT NULL
            ORDER BY stay_id, charttime, itemid
        ) TO '{output_dir}/labs_ce.csv' (DELIMITER '{sep}', HEADER);
    """)

    # mechvent is 1 if any mechanical ventilation chartevent is found at that time.
    # itemid 223894 is excluded when value = 'Other/Remarks' (non-ventilator use).
    con.execute(f"""
        COPY (
            SELECT stay_id, charttime,
                MAX(CASE
                    WHEN itemid = 223894 AND value != 'Other/Remarks' THEN 1
                    WHEN itemid = 226732 AND value = 'Ventilator' THEN 1
                    WHEN itemid IN ({csv_list([x for x in MECHVENT_ITEMIDS if x not in (223894, 226732)])}) THEN 1
                    ELSE 0
                END) AS mechvent
            FROM ce_filtered
            WHERE itemid IN ({csv_list(MECHVENT_ITEMIDS)})
              AND stay_id IS NOT NULL
              AND value IS NOT NULL
            GROUP BY stay_id, charttime
        ) TO '{output_dir}/mechvent.csv' (DELIMITER '{sep}', HEADER);
    """)

    # GCS requires all three components (eye=220739, verbal=223900, motor=223901)
    # to be present at the same charttime; rows with any component missing are dropped.
    con.execute(f"""
        COPY (
            SELECT stay_id, charttime, (gcs_eye + gcs_verbal + gcs_motor) AS gcs
            FROM (
                SELECT stay_id, charttime,
                    MAX(CASE WHEN itemid = 220739 THEN valuenum END) AS gcs_eye,
                    MAX(CASE WHEN itemid = 223900 THEN valuenum END) AS gcs_verbal,
                    MAX(CASE WHEN itemid = 223901 THEN valuenum END) AS gcs_motor
                FROM ce_filtered
                WHERE itemid IN ({csv_list(GCS_ITEMIDS)})
                  AND stay_id IS NOT NULL
                GROUP BY stay_id, charttime
            )
            WHERE gcs_eye IS NOT NULL
              AND gcs_verbal IS NOT NULL
              AND gcs_motor IS NOT NULL
            ORDER BY stay_id, charttime
        ) TO '{output_dir}/gcs.csv' (DELIMITER '{sep}', HEADER);
    """)
    con.execute("DROP TABLE IF EXISTS ce_filtered")


@timed("Extract antibiotics")
def extract_abx(con, paths: MIMICPaths, output_dir: str, sep: str) -> None:
    """
    Extract antibiotic prescription rows from prescriptions.csv, filtering by
    the GSN allowlist.  Only start/stop timestamps are kept; the drug name is
    retained for downstream labelling but is NOT used as a feature (leakage
    prevention).
    """
    con.execute(f"""
        COPY (
            SELECT
                subject_id,
                hadm_id,
                drug,
                epoch(starttime::TIMESTAMP) AS starttime,
                epoch(stoptime::TIMESTAMP) AS stoptime
            FROM read_csv_auto('{paths.h("prescriptions.csv")}')
            WHERE gsn IN ({sql_string_list(ANTIBIOTIC_GSN_CODES)})
            ORDER BY subject_id, hadm_id
        ) TO '{output_dir}/abx.csv' (DELIMITER '{sep}', HEADER);
    """)


@timed("Extract microbiology timestamps")
def extract_microbio(con, paths: MIMICPaths, output_dir: str, sep: str) -> None:
    """
    Extract charttime and chartdate from microbiologyevents.  Both columns are
    kept because some rows have chartdate but no charttime; they are merged
    later in run_onset_generation via merge_microbiology().
    """
    con.execute(f"""
        COPY (
            SELECT
                subject_id,
                hadm_id,
                epoch(charttime::TIMESTAMP) AS charttime,
                epoch(chartdate::TIMESTAMP) AS chartdate
            FROM read_csv_auto('{paths.h("microbiologyevents.csv")}')
        ) TO '{output_dir}/microbio.csv' (DELIMITER '{sep}', HEADER);
    """)


@timed("Extract hospital labs")
def extract_labs_le(con, paths: MIMICPaths, output_dir: str, sep: str, mapping_file: str) -> None:
    """
    Extract lab results from the hospital labevents table.  Each row is joined
    to the matching ICU stay via hadm_id; only events within ±1 day of the
    ICU window are kept to avoid pulling unrelated hospitalisation lab data.
    """
    _, _, lab_codes = load_mapping_codes(mapping_file)
    con.execute(f"""
        COPY (
            SELECT
                i.stay_id,
                epoch(le.charttime::TIMESTAMP) AS timestp,
                le.itemid,
                le.valuenum
            FROM read_csv_auto('{paths.i("icustays.csv")}') i
            INNER JOIN read_csv_auto('{paths.h("labevents.csv")}') le
                ON le.hadm_id = i.hadm_id
                AND le.charttime::TIMESTAMP >= i.intime::TIMESTAMP - INTERVAL '1 day'
                AND le.charttime::TIMESTAMP <= i.outtime::TIMESTAMP + INTERVAL '1 day'
                AND le.itemid IN ({csv_list(lab_codes)})
                AND le.valuenum IS NOT NULL
            ORDER BY i.stay_id, le.charttime, le.itemid
        ) TO '{output_dir}/labs_le.csv' (DELIMITER '{sep}', HEADER);
    """)


@timed("Extract fluids and vasopressors")
def extract_inputevents_all(con, paths: MIMICPaths, output_dir: str, sep: str) -> None:
    """
    Single pass over inputevents.csv to produce three output files:
    - vaso.csv         : vasopressor infusions with rate normalised to norepinephrine
                         equivalents (mcg/kg/min) using published dose-conversion factors.
    - fluid.csv        : IV fluid administrations with `tev` (tonicity-equivalent volume)
                         rescaling colloids/hypertonics to an effective crystalloid equivalent.
    - preadm_fluid.csv : total pre-admission fluid as a single scalar per stay.

    Vasopressor normalisation (CASE block):
    - Norepinephrine (221906, 221289): 1:1 if already mcg/kg/min, ÷80 for mcg/min.
    - Vasopressin (222315): converted to norepinephrine equivalent via a 5 unit/h
      = 0.04 mcg/kg/min approximation (typical dose-equivalence).
    - Phenylephrine (221749): multiply by 0.45 (published NE-equivalent).
    - Dopamine (221662): multiply by 0.01 (published NE-equivalent).
    Rows with statusdescription='Rewritten' (cancelled duplicates) are excluded.

    Fluid tev multipliers reflect tonicity-equivalent volumes per mL:
    - Albumin 25% (225823, 225159): ×0.5
    - D5W/electrolyte (227531): ×2.75
    - Hypertonic saline (225161): ×3
    - 25% albumin (220862): ×5
    - Dextrose/NaCl solutions (220995, 227533): ×6.66
    - 3% NaCl (228341): ×8
    """
    all_itemids = sorted(set(VASO_ITEMIDS) | set(FLUID_ITEMIDS) | set(PREADM_FLUID_ITEMIDS))
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ie_filtered AS
        SELECT
            stay_id,
            itemid,
            amount,
            rate,
            rateuom,
            epoch(starttime::TIMESTAMP) AS starttime,
            epoch(endtime::TIMESTAMP) AS endtime,
            statusdescription
        FROM read_csv_auto('{paths.i("inputevents.csv")}')
        WHERE itemid IN ({csv_list(all_itemids)})
    """)

    con.execute(f"""
        COPY (
            SELECT stay_id, itemid, starttime, endtime,
                CASE
                    WHEN itemid IN (221906, 221289) AND rateuom = 'mcg/kg/min' THEN ROUND(rate::NUMERIC, 3)
                    WHEN itemid IN (221906, 221289) AND rateuom = 'mcg/min' THEN ROUND((rate / 80)::NUMERIC, 3)
                    WHEN itemid = 222315 AND rate > 0.2 THEN ROUND((rate * 5 / 60)::NUMERIC, 3)
                    WHEN itemid = 222315 AND rateuom = 'units/min' THEN ROUND((rate * 5)::NUMERIC, 3)
                    WHEN itemid = 222315 AND rateuom = 'units/hour' THEN ROUND((rate * 5 / 60)::NUMERIC, 3)
                    WHEN itemid = 221749 AND rateuom = 'mcg/kg/min' THEN ROUND((rate * 0.45)::NUMERIC, 3)
                    WHEN itemid = 221749 AND rateuom = 'mcg/min' THEN ROUND((rate * 0.45 / 80)::NUMERIC, 3)
                    WHEN itemid = 221662 AND rateuom = 'mcg/kg/min' THEN ROUND((rate * 0.01)::NUMERIC, 3)
                    WHEN itemid = 221662 AND rateuom = 'mcg/min' THEN ROUND((rate * 0.01 / 80)::NUMERIC, 3)
                    ELSE NULL
                END AS rate_std
            FROM ie_filtered
            WHERE itemid IN ({csv_list(VASO_ITEMIDS)})
              AND rate IS NOT NULL
              AND statusdescription <> 'Rewritten'
            ORDER BY stay_id, itemid, starttime
        ) TO '{output_dir}/vaso.csv' (DELIMITER '{sep}', HEADER);
    """)

    con.execute(f"""
        COPY (
            SELECT stay_id, starttime, endtime, itemid,
                ROUND(amount::NUMERIC, 3) AS amount,
                ROUND(rate::NUMERIC, 3) AS rate,
                ROUND((CASE
                    WHEN itemid IN (225823, 225159) THEN amount * 0.5
                    WHEN itemid = 227531 THEN amount * 2.75
                    WHEN itemid = 225161 THEN amount * 3
                    WHEN itemid = 220862 THEN amount * 5
                    WHEN itemid IN (220995, 227533) THEN amount * 6.66
                    WHEN itemid = 228341 THEN amount * 8
                    ELSE amount
                END)::NUMERIC, 3) AS tev
            FROM ie_filtered
            WHERE itemid IN ({csv_list(FLUID_ITEMIDS)})
              AND stay_id IS NOT NULL
              AND amount IS NOT NULL
            ORDER BY stay_id, starttime, itemid
        ) TO '{output_dir}/fluid.csv' (DELIMITER '{sep}', HEADER);
    """)

    con.execute(f"""
        COPY (
            SELECT stay_id, SUM(amount) AS inputpreadm
            FROM ie_filtered
            WHERE itemid IN ({csv_list(PREADM_FLUID_ITEMIDS)})
            GROUP BY stay_id
        ) TO '{output_dir}/preadm_fluid.csv' (DELIMITER '{sep}', HEADER);
    """)
    con.execute("DROP TABLE IF EXISTS ie_filtered")


@timed("Extract urine output")
def extract_uo(con, paths: MIMICPaths, output_dir: str, sep: str) -> None:
    """Extract urine output measurements from outputevents."""
    con.execute(f"""
        COPY (
            SELECT stay_id, epoch(charttime::TIMESTAMP) AS charttime, itemid, value
            FROM read_csv_auto('{paths.i("outputevents.csv")}')
            WHERE stay_id IS NOT NULL
              AND value IS NOT NULL
              AND itemid IN ({csv_list(UO_ITEMIDS)})
            ORDER BY stay_id, charttime, itemid
        ) TO '{output_dir}/uo.csv' (DELIMITER '{sep}', HEADER);
    """)


def run_extraction(args: argparse.Namespace) -> None:
    """
    Run the full DuckDB extraction pass over raw MIMIC-IV CSV files.
    Sets 8 GB memory limit and 4 threads for the DuckDB in-process engine.
    """
    os.makedirs(args.extracted_dir, exist_ok=True)
    paths = MIMICPaths(args.mimic_dir)
    ensure_inputs(paths)

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB';")
    con.execute("SET threads TO 4;")
    try:
        extract_icustays(con, paths, args.extracted_dir, args.separator)
        extract_demog(con, paths, args.extracted_dir, args.separator)
        extract_chartevents_all(con, paths, args.extracted_dir, args.separator, args.mapping_file)
        extract_abx(con, paths, args.extracted_dir, args.separator)
        extract_microbio(con, paths, args.extracted_dir, args.separator)
        extract_labs_le(con, paths, args.extracted_dir, args.separator, args.mapping_file)
        extract_inputevents_all(con, paths, args.extracted_dir, args.separator)
        extract_uo(con, paths, args.extracted_dir, args.separator)
    finally:
        con.close()


def read_processed(output_dir: str, filename: str) -> pd.DataFrame:
    """Read a pipe-delimited CSV from a stage directory."""
    return pd.read_csv(os.path.join(output_dir, filename), sep="|")


def determine_readmission(demog: pd.DataFrame) -> pd.DataFrame:
    """
    Flag re-admissions: re_admission=1 when the previous hospital discharge
    was within 30 days of this admission (readmission within 30 days).
    All timestamps are Unix epoch seconds.
    """
    demog = demog.sort_values(["subject_id", "admittime"]).copy()
    demog["previous_dischtime"] = demog.groupby("subject_id")["dischtime"].shift(1)
    gap = demog["admittime"] - demog["previous_dischtime"]
    demog["re_admission"] = ((gap >= 0) & (gap <= 30 * 24 * 3600)).astype(int)
    return demog.drop(columns=["previous_dischtime"])


def merge_microbiology(microbio: pd.DataFrame, culture: pd.DataFrame) -> pd.DataFrame:
    """
    Combine microbiologyevents (charttime; chartdate fallback) with culture
    chartevents rows into one unified bacteriology table.  Using chartdate
    as fallback adds midnight timestamps for samples that lack an exact time.
    """
    microbio = microbio.copy()
    microbio["charttime"] = microbio["charttime"].fillna(microbio["chartdate"])
    microbio = microbio.drop(columns=["chartdate"], errors="ignore")
    return pd.concat([microbio, culture], sort=False, ignore_index=True)


def fill_stay_ids(events: pd.DataFrame, demog: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """
    Back-fill missing stay_id values in an event table using the hadm_id and
    event timestamp to find the matching ICU stay in `demog`.

    The tolerance window is ±48 h around ICU in/out times to catch events
    charted slightly before ICU admission or after discharge.  If only one
    stay exists for the admission, it is assigned regardless of timing.
    """
    events = events.copy()
    if "stay_id" not in events.columns:
        events["stay_id"] = np.nan

    missing = events["stay_id"].isna()
    if not missing.any():
        return events

    demog_by_hadm = {hadm_id: group for hadm_id, group in demog.groupby("hadm_id")}
    for idx, row in events[missing].iterrows():
        hadm_id = row.get("hadm_id")
        event_time = row.get(time_column)
        candidates = demog_by_hadm.get(hadm_id)
        if candidates is None or pd.isna(event_time):
            continue

        window = candidates[
            (event_time >= candidates["intime"] - 48 * 3600)
            & (event_time <= candidates["outtime"] + 48 * 3600)
        ]
        if len(window) > 0:
            events.at[idx, "stay_id"] = window.iloc[0]["stay_id"]
        elif len(candidates) == 1:
            events.at[idx, "stay_id"] = candidates.iloc[0]["stay_id"]

    return events


def find_infection_onset(abx: pd.DataFrame, bacterio: pd.DataFrame, lead_time: int) -> pd.DataFrame:
    """
    LEGACY -- superseded by Stage 2. Kept only because it feeds `onset.csv`, which
    nothing downstream reads; see `run_onset_generation` for the full note.
    The label the project actually uses comes from
    `3_build_sepsis3_labels.py::find_suspected_infection`.

    Identify suspected infection onset for each ICU stay using the Sepsis-3
    antibiotic/culture co-occurrence rule:

      Forward rule: antibiotic administered AND culture collected within 24 h after.
      Backward rule: culture collected AND antibiotic administered within 72 h after.

    The onset_time is the earliest trigger time (antibiotic for forward,
    culture for backward).  prediction_time = onset_time - lead_time hours.

    Returns one row per stay with: subject_id, stay_id, onset_time,
    prediction_time, label=1.  Stays without a qualifying pair are omitted.
    """
    onset_rows = []
    abx_by_stay = {stay_id: group.sort_values("starttime") for stay_id, group in abx.dropna(subset=["stay_id"]).groupby("stay_id")}
    bact_by_stay = {
        stay_id: group.sort_values("charttime")
        for stay_id, group in bacterio.dropna(subset=["stay_id", "charttime"]).groupby("stay_id")
    }

    for stay_id in sorted(set(abx_by_stay) & set(bact_by_stay)):
        abx_times = abx_by_stay[stay_id]["starttime"].dropna().to_numpy()
        bact_times = bact_by_stay[stay_id]["charttime"].dropna().to_numpy()
        if len(abx_times) == 0 or len(bact_times) == 0:
            continue

        subject_id = bact_by_stay[stay_id]["subject_id"].dropna()
        subject_id = subject_id.iloc[0] if len(subject_id) else np.nan

        found = None
        for abx_time in abx_times:
            differences_hours = (bact_times - abx_time) / 3600
            # Forward check: culture ≤24 h after antibiotic
            valid_after_abx = differences_hours[(differences_hours >= 0) & (differences_hours <= 24)]
            if len(valid_after_abx) > 0:
                found = abx_time
                break

            # Backward check: culture ≤72 h before antibiotic
            valid_before_abx = differences_hours[(differences_hours <= 0) & (differences_hours >= -72)]
            if len(valid_before_abx) > 0:
                found = bact_times[np.where(differences_hours == valid_before_abx.max())[0][0]]
                break

        if found is not None:
            onset_rows.append(
                {
                    "subject_id": subject_id,
                    "stay_id": stay_id,
                    "onset_time": found,
                    "prediction_time": found - lead_time * 3600,
                    "label": 1,
                }
            )

    return pd.DataFrame(onset_rows)


def build_controls(demog: pd.DataFrame, positive_onset: pd.DataFrame, control_ratio: float) -> pd.DataFrame:
    """
    LEGACY -- do not copy this sampling scheme. It clips the drawn offset to the
    control's ICU discharge time, which is exactly the design flaw that inflated the
    primary AUROC from 0.721 to 0.838: 32.3% of controls ended up with
    `prediction_time == outtime` and their 24h observation window described a
    pre-discharge stabilisation rather than the run-up to a deterioration. The
    correct, offset-matched implementation is
    `3_build_sepsis3_labels.py::build_controls`; see "Control Construction" in
    `SEPSIS3_TASK_DECISIONS.md`.

    This copy survives only because it feeds `onset.csv`, which nothing downstream
    reads.

    Sample temporally-matched negative (control) stays.

    Each control's pseudo-prediction_time is drawn from the empirical
    distribution of (prediction_time - admittime) offsets of the positive
    cases, then clipped to the stay's discharge time.

    control_ratio is capped at 1.0 to maintain class balance.
    """
    if positive_onset.empty:
        return pd.DataFrame(columns=["subject_id", "stay_id", "onset_time", "prediction_time", "label"])

    positive_stays = set(positive_onset["stay_id"])
    controls = demog[~demog["stay_id"].isin(positive_stays)].copy()
    if controls.empty:
        return pd.DataFrame(columns=["subject_id", "stay_id", "onset_time", "prediction_time", "label"])

    pos_with_admit = positive_onset.merge(demog[["stay_id", "admittime"]], on="stay_id", how="left")
    offsets = (pos_with_admit["prediction_time"] - pos_with_admit["admittime"]).dropna()
    offsets = offsets[offsets > 0].to_numpy()
    if len(offsets) == 0:
        # Fallback offsets in seconds when the positive distribution is empty.
        offsets = np.array([12, 24, 36, 48]) * 3600

    n_controls = int(min(len(controls), len(positive_onset) * min(control_ratio, 1.0)))
    controls = controls.sample(n=n_controls, random_state=42) if n_controls < len(controls) else controls

    rng = np.random.default_rng(42)
    sampled_offsets = rng.choice(offsets, size=len(controls), replace=True)
    controls["prediction_time"] = controls["admittime"] + sampled_offsets
    controls["prediction_time"] = controls[["prediction_time", "dischtime"]].min(axis=1)
    controls["onset_time"] = controls["prediction_time"]
    controls["label"] = 0
    return controls[["subject_id", "stay_id", "onset_time", "prediction_time", "label"]]


@timed("Build onset labels and processed helper files")
def run_onset_generation(args: argparse.Namespace) -> None:
    """
    Build onset.csv (suspected-infection timing, not the final Sepsis-3 label)
    and several processed helper files used by downstream stages.

    `onset.csv` is a LEGACY artifact: Stage 2 (`3_build_sepsis3_labels.py`) recomputes
    both the suspected-infection timing and the control arm from scratch into
    `sepsis3_onset.csv`, and that is what every later stage reads. `onset.csv` is still
    written so the Stage-1b audit (`2_audit_inputs.py`) keeps a column contract that
    fails loudly on a truncated extraction. Consequently `--lead_time` and
    `--control_ratio` on this stage affect only that file, never the final label.

    The helper files below, by contrast, ARE consumed by Stage 2 and are not affected
    by those two flags.

    - onset.csv              : LEGACY. positives (label=1) + controls (label=0)
    - bacterio_processed.csv : unified microbio + culture events with stay_ids filled
    - demog_processed.csv    : demographics with re_admission flag
    - labu.csv               : concatenation of labs_ce (bedside) + labs_le (hospital)
    - abx_processed.csv      : antibiotics with stay_ids filled

    The fluid.csv is also augmented with `norm_rate_of_infusion`
    (tev × rate / amount) when the tev, rate, and amount columns are present.
    """
    os.makedirs(args.onset_dir, exist_ok=True)
    demog = read_processed(args.extracted_dir, "demog.csv")
    abx = read_processed(args.extracted_dir, "abx.csv")
    culture = read_processed(args.extracted_dir, "culture.csv")
    microbio = read_processed(args.extracted_dir, "microbio.csv")
    labs_ce = read_processed(args.extracted_dir, "labs_ce.csv")
    labs_le = read_processed(args.extracted_dir, "labs_le.csv")
    fluid = read_processed(args.extracted_dir, "fluid.csv")

    demog = determine_readmission(demog)
    bacterio = merge_microbiology(microbio, culture)
    bacterio = fill_stay_ids(bacterio, demog, "charttime")
    abx = fill_stay_ids(abx, demog, "starttime")

    # Unify lab sources: rename labs_le timestamp column and stack both tables.
    labs_le = labs_le.rename(columns={"timestp": "charttime"})
    labu = pd.concat([labs_ce, labs_le], sort=False, ignore_index=True)

    if {"tev", "rate", "amount"}.issubset(fluid.columns):
        nonzero = fluid["amount"] != 0
        fluid.loc[nonzero, "norm_rate_of_infusion"] = (
            fluid.loc[nonzero, "tev"] * fluid.loc[nonzero, "rate"] / fluid.loc[nonzero, "amount"]
        )
        fluid.to_csv(os.path.join(args.onset_dir, "fluid.csv"), sep=args.separator, index=False)

    positive_onset = find_infection_onset(abx, bacterio, args.lead_time)
    controls = build_controls(demog, positive_onset, args.control_ratio)
    onset = pd.concat([positive_onset, controls], ignore_index=True)

    onset.to_csv(os.path.join(args.onset_dir, "onset.csv"), sep=args.separator, index=False)
    bacterio.to_csv(os.path.join(args.onset_dir, "bacterio_processed.csv"), sep=args.separator, index=False)
    demog.to_csv(os.path.join(args.onset_dir, "demog_processed.csv"), sep=args.separator, index=False)
    labu.to_csv(os.path.join(args.onset_dir, "labu.csv"), sep=args.separator, index=False)
    abx.to_csv(os.path.join(args.onset_dir, "abx_processed.csv"), sep=args.separator, index=False)

    print(f"Positive sepsis stays: {len(positive_onset)}")
    print(f"Control stays:         {len(controls)}")
    print(f"Total onset rows:      {len(onset)}")


@timed("Re-extract demographics only")
def run_demog_only(args: argparse.Namespace) -> None:
    """Rebuild demog.csv and demog_processed.csv, leaving every other artifact alone.

    `demog_processed.csv` is normally a by-product of `run_onset_generation`,
    which also rewrites onset.csv, bacterio_processed.csv, labu.csv and
    abx_processed.csv -- the last of which means re-stacking the two lab tables
    (763 MB) for nothing. The only transformation demographics receive there is
    `determine_readmission`, so it is reproduced here in isolation.
    """
    os.makedirs(args.extracted_dir, exist_ok=True)
    os.makedirs(args.onset_dir, exist_ok=True)
    paths = MIMICPaths(args.mimic_dir)

    con = duckdb.connect()
    con.execute("SET memory_limit = '8GB';")
    con.execute("SET threads TO 4;")
    try:
        extract_demog(con, paths, args.extracted_dir, args.separator)
    finally:
        con.close()

    demog = read_processed(args.extracted_dir, "demog.csv")
    demog = determine_readmission(demog)
    demog.to_csv(os.path.join(args.onset_dir, "demog_processed.csv"), sep=args.separator, index=False)
    print(f"ICU stays: {len(demog)} | columns: {len(demog.columns)}")
    print(f"Wrote {os.path.join(args.extracted_dir, 'demog.csv')}")
    print(f"Wrote {os.path.join(args.onset_dir, 'demog_processed.csv')}")


def main() -> None:
    args = parse_args()
    if args.only_demog:
        run_demog_only(args)
        return
    if not args.skip_extraction:
        run_extraction(args)
    if not args.skip_onset:
        run_onset_generation(args)


if __name__ == "__main__":
    main()
