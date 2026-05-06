# utils/data_pipeline.py

import os
import sys
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from copy import deepcopy
from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default DATA_DIR (relative to the project root, which is 5 levels up from utils/)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../data/'))

class CohortExtractor:
    def __init__(self, db_config: Optional[dict] = None):
        self.db_config = db_config
        self.conn = None
        self.data_dir = DATA_DIR

        if db_config:
            import psycopg2

            try:
                self.conn = psycopg2.connect(**db_config)
                print("✅ Connected to MIMIC-III database.")
            except Exception as e:
                raise RuntimeError(f"❌ Database connection failed: {e}")
        else:
            print("No DB config provided. Using CSV mode.")

    def extract_cohort(self) -> pd.DataFrame:
        if self.conn:
            df = pd.read_sql("SELECT * FROM mimiciii.icustays LIMIT 10", self.conn)
            if df.empty:
                raise ValueError("⚠️ Query returned no data.")
            return df

        # CSV MODE

        required_files = ['ADMISSIONS.csv', 'PATIENTS.csv', 'ICUSTAYS.csv', 'DIAGNOSES_ICD.csv']
        for f in required_files:
            path = os.path.join(self.data_dir, f)
            if not os.path.exists(path):
                raise FileNotFoundError(f"❌ Required file not found: {path}")

        admissions = pd.read_csv(os.path.join(self.data_dir, 'ADMISSIONS.csv'))
        patients   = pd.read_csv(os.path.join(self.data_dir, 'PATIENTS.csv'))
        icustays   = pd.read_csv(os.path.join(self.data_dir, 'ICUSTAYS.csv'))
        diagnoses  = pd.read_csv(os.path.join(self.data_dir, 'DIAGNOSES_ICD.csv'))

        if any(df.empty for df in [admissions, patients, icustays, diagnoses]):
            raise ValueError("⚠️ One or more input CSV files are empty.")

        # 1. Explicit sepsis codes
        explicit_sepsis_codes = ['038%', '995.91', '995.92', '785.52']
        explicit_mask = diagnoses['icd9_code'].astype(str).str.startswith(
            tuple(c.replace('%', '') for c in explicit_sepsis_codes)
        )

        # 2. Implicit sepsis: infection + organ dysfunction
        infection_codes = [
            '001', '002', '003', '004', '005', '008', '009', '010', '011', '012', '013', '014', '015', '016',
            '017', '018', '020', '021', '022', '023', '024', '025', '026', '027', '030', '031', '032', '033',
            '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047',
            '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061',
            '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075',
            '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089',
            '090', '091', '092', '093', '094', '095', '096', '097', '098', '099', '100', '101', '102', '103',
            '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117',
            '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131',
            '132', '133', '134', '135', '136', '137', '138', '139', '460', '461', '462', '463', '464', '465',
            '466', '480', '481', '482', '483', '484', '485', '486', '487', '488', '507', '510', '511', '513',
            '5180', '5190', '590', '595', '5990', '601', '614', '615', '616', '681', '682', '711', '730', '7907',
            '9966', '9985'
        ]

        organ_dysfunction_codes = [
            # Cardiovascular
            '7855', '458', '4580', '4589', '7962',
            # Respiratory
            '51881', '51882', '51884', '5185', '7991',
            # Renal
            '580', '581', '582', '583', '584', '585', '586', '403', '404',
            # Hepatic
            '570', '5722', '5723', '5724', '5728',
            # Hematologic
            '286', '2860', '2861', '2862', '2866', '2869', '287',
            # Neurologic
            '3483', '34831', '34832', '78001', '78003', '78009'
        ]

        # Explicit sepsis
        explicit = diagnoses[diagnoses['icd9_code'].astype(str).str.startswith(tuple(explicit_sepsis_codes))]

        # Implicit sepsis: infection + at least one organ dysfunction
        infection = diagnoses[diagnoses['icd9_code'].astype(str).str.startswith(tuple(infection_codes))]
        organ_dys = diagnoses[diagnoses['icd9_code'].astype(str).str.startswith(tuple(organ_dysfunction_codes))]

        implicit_infection = infection['hadm_id'].unique()
        implicit_organ = organ_dys['hadm_id'].unique()

        implicit = set(implicit_infection) & set(implicit_organ)

        # Combine explicit and implicit
        sepsis_hadm_ids = set(explicit['hadm_id'].unique()) | implicit

        print(f"Angus sepsis cases identified: {len(sepsis_hadm_ids)} hospital admissions")

        # Build cohort
        cohort = icustays[icustays['hadm_id'].isin(sepsis_hadm_ids)].copy()

        # Merge patients and admissions info
        cohort = pd.merge(cohort, patients, 
                          on='subject_id', how='left')
        cohort = pd.merge(cohort, admissions[['hadm_id', 'hospital_expire_flag']], 
                          on='hadm_id', how='left')

        # Feature engineering
        cohort['intime'] = pd.to_datetime(cohort['intime'])
        cohort['outtime'] = pd.to_datetime(cohort['outtime'])
        cohort['dob'] = pd.to_datetime(cohort['dob'])

        cohort['age'] = (cohort['intime'].dt.year - cohort['dob'].dt.year).clip(upper=89)
        cohort['los_hours'] = ((cohort['outtime'] - cohort['intime']).dt.total_seconds() / 3600)

        cohort['mortality_hosp'] = cohort['hospital_expire_flag'].fillna(0)
        cohort['mortality_48h'] = ((cohort['hospital_expire_flag'] == 1) & (cohort['los_hours'] <= 48)).astype(int)

        # Final filters (adult + reasonable ICU stay)
        cohort = cohort[
            (cohort['age'] >= 18) &
            (cohort['los_hours'] >= 24) &
            (cohort['los_hours'] <= 168)
        ].copy()

        if cohort.empty:
            raise ValueError("Cohort is empty after filtering.")

        print(f"✅ Extracted {len(cohort):,} Angus sepsis trajectories (adult, LOS 24-168h)")

        return cohort.sort_values(['subject_id', 'intime']).reset_index(drop=True)


class FeatureConfig:

    vitals: List[str] = ('heart_rate', 'sbp', 'dbp', 'map', 
                         'resp_rate', 'spo2', 'temperature', 'gcs')
    labs: List[str] = ('wbc', 'hemoglobin', 'platelets', 'creatinine', 
                       'bun', 'lactate', 'ph', 'pao2', 'pco2', 'bicarbonate', 
                       'anion_gap', 'glucose', 'bilirubin')
    derived: List[str] = ('sofa_score', 'sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_renal', 
                          'sirs_score', 'shock_index', 'pf_ratio', 'urine_output_6h')
    static: List[str] = ('age', 'gender', 'weight')
    contextual: List[str] = ('hour_of_day', 'icu_hour', 'ventilation_status', 
                             'dialysis_status', 'cumulative_fluid_24h')


class FeatureExtractor:
    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.config = config
        self.data_dir = DATA_DIR
 
    def _load_csv(self, name):
        path = os.path.join(self.data_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Missing required file: {path}")
        return pd.read_csv(path)

    def _extract_static_features(self, icustay_id: int) -> dict:

        try:
            patients = self._load_csv('PATIENTS.csv')
            icustays = self._load_csv('ICUSTAYS.csv')
            chartevents = self._load_csv('CHARTEVENTS.csv')
        except FileNotFoundError as e:
            print(f"⚠️  CSV file not found: {e}")
            # Return default values if files missing
            return {
                'age': 65.0,
                'gender': 1.0,
                'weight': 75.0
            }
 
        icu_row = icustays[icustays['icustay_id'] == icustay_id]
        if icu_row.empty:
            print(f"⚠️  ICU stay {icustay_id} not found. Using defaults.")
            return {
                'age': 65.0,
                'gender': 1.0,
                'weight': 75.0
            }
 
        subject_id = icu_row.iloc[0]['subject_id']
        intime = pd.to_datetime(icu_row.iloc[0]['intime'])
 
        patient = patients[patients['subject_id'] == subject_id]
        if patient.empty:
            print(f"⚠️  Patient {subject_id} not found. Using defaults.")
            return {
                'age': 65.0,
                'gender': 1.0,
                'weight': 75.0
            }
 
        dob = pd.to_datetime(patient.iloc[0]['dob'])
        age = min(intime.year - dob.year, 89)
        gender = 1 if patient.iloc[0]['gender'] == 'M' else 0
 
        # Weight (first ICU measurement) 
        weight_ids = [762, 763, 224639, 226512]
        ce_w = chartevents[
            (chartevents['icustay_id'] == icustay_id) &
            (chartevents['itemid'].isin(weight_ids))
        ].copy()

        weight = np.nan
        if not ce_w.empty:
            ce_w['charttime'] = pd.to_datetime(ce_w['charttime'])
            ce_w = ce_w.sort_values('charttime')
            weight_series = ce_w['valuenum'].dropna()
            if len(weight_series) > 0:
                weight = float(weight_series.iloc[0])

        return {
            'age': float(age),
            'gender': float(gender),
            'weight': float(weight)
        }

    def extract_hourly_features(
        self,
        icustay_id: int,
        intime: pd.Timestamp,
        outtime: pd.Timestamp
    ) -> pd.DataFrame:
 
        hours = pd.date_range(intime, outtime, freq='1H')
        df = pd.DataFrame(index=hours)
 
        try:
            chartevents = self._load_csv('CHARTEVENTS.csv')
            labevents = self._load_csv('LABEVENTS.csv')
            outputevents = self._load_csv('OUTPUTEVENTS.csv')
        except FileNotFoundError as e:
            print(f"⚠️  CSV file not found: {e}")
            # Return minimal dataframe with defaults
            return self._create_default_features(hours)
 
        chartevents = chartevents[chartevents['icustay_id'] == icustay_id]
 
        hadm_ids = chartevents['hadm_id'].dropna().unique()
        if len(hadm_ids) > 0:
            labevents = labevents[labevents['hadm_id'].isin(hadm_ids)]
        else:
            labevents = pd.DataFrame()
 
        chartevents['charttime'] = pd.to_datetime(chartevents['charttime'])
        if not labevents.empty:
            labevents['charttime'] = pd.to_datetime(labevents['charttime'])
        if not outputevents.empty:
            outputevents['charttime'] = pd.to_datetime(outputevents['charttime'])
 
        # -------------------------
        # ITEMID MAP
        # -------------------------
        ITEM_MAP = {
            'heart_rate': [211, 220045],
            'sbp': [51, 442, 455, 220179],
            'dbp': [8368, 8440, 220180],
            'map': [456, 52, 220181],
            'resp_rate': [618, 220210],
            'spo2': [646, 220277],
            'temperature': [223761, 678],
            'gcs': [198, 226755],
 
            'pao2': [50821],
            'pco2': [50818],
            'ph': [50820],
            'bicarbonate': [50882],
            'lactate': [50813],
            'wbc': [51300],
            'hemoglobin': [51222],
            'platelets': [51265],
            'creatinine': [50912],
            'bun': [51006],
            'glucose': [50931, 50809],
            'bilirubin': [50885],
        }
 
        def extract_series(events, itemids, agg='median'):
            sub = events[events['itemid'].isin(itemids)].copy()
            sub = sub[(sub['charttime'] >= intime) & (sub['charttime'] <= outtime)]

            if sub.empty:
                return pd.Series(index=hours, dtype=float)

            sub = sub.set_index('charttime')

            # Handle different column names
            value_col = 'value' if 'value' in sub.columns else 'valuenum'
            
            sub[value_col] = pd.to_numeric(sub[value_col], errors='coerce')
            
            if agg == 'median':
                hourly = sub[value_col].resample('1h').median()
            else:
                hourly = sub[value_col].resample('1h').sum()

            return hourly.reindex(hours)

        # Vitals from CHARTEVENTS
        for v in self.config.vitals:
            if v in ITEM_MAP:
                df[v] = extract_series(chartevents, ITEM_MAP[v], agg='median')

        # LABS (FFILL ≤6h)
        
        if not labevents.empty:
            for l in self.config.labs:
                if l in ITEM_MAP:
                    df[l] = extract_series(labevents, ITEM_MAP[l]).ffill(limit=6)
        else:
            # No labs available
            for l in self.config.labs:
                df[l] = np.nan

        # Urine output from OUTPUTEVENTS (uses 'value')
        if not outputevents.empty:
            urine_ids = [40055, 43175, 40069]
            urine = extract_series(outputevents, urine_ids, agg='sum')
            df['urine_output_6h'] = urine.rolling(6).sum()
        else:
            df['urine_output_6h'] = np.nan

        # Derived features
        df['shock_index'] = df['heart_rate'] / df['sbp'].replace(0, np.nan)
        
        fio2_ids = [190, 223835]
        fio2 = extract_series(chartevents, fio2_ids, agg='median')
        df['pf_ratio'] = df['pao2'] / fio2.where(fio2 > 0, np.nan)
        
        df_temp = df.copy()
        
        vital_lab_cols = ['pf_ratio', 'platelets', 'bilirubin', 'creatinine', 
                         'heart_rate', 'sbp', 'temperature', 'wbc']
        df_temp[vital_lab_cols] = df_temp[vital_lab_cols].ffill(limit=6).fillna(0)

        # SOFA components
        df['sofa_resp'] = np.where(df_temp['pf_ratio'] < 300, 2, 
                          np.where(df_temp['pf_ratio'] < 400, 1, 0))
        df['sofa_coag'] = np.where(df_temp['platelets'] < 50, 3, 
                          np.where(df_temp['platelets'] < 100, 2, 
                          np.where(df_temp['platelets'] < 150, 1, 0)))
        df['sofa_liver'] = np.where(df_temp['bilirubin'] >= 12, 4, 
                           np.where(df_temp['bilirubin'] >= 6, 3, 
                           np.where(df_temp['bilirubin'] >= 2, 2, 
                           np.where(df_temp['bilirubin'] >= 1.2, 1, 0))))
        df['sofa_renal'] = np.where(df_temp['creatinine'] >= 5, 4, 
                           np.where(df_temp['creatinine'] >= 3.5, 3, 
                           np.where(df_temp['creatinine'] >= 2, 2, 
                           np.where(df_temp['creatinine'] >= 1.2, 1, 0))))

        df['sofa_score'] = df[['sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_renal']].fillna(0).sum(axis=1)

        df['sirs_score'] = (
            (df_temp['temperature'] > 38).astype(int) +
            (df_temp['temperature'] < 36).astype(int) +
            (df_temp['heart_rate'] > 90).astype(int) +
            (df['resp_rate'] > 20).astype(int) +   
            (df_temp['wbc'] > 12).astype(int) +
            (df_temp['wbc'] < 4).astype(int)
        )

        # Contextual features
        df['hour_of_day'] = df.index.hour
        df['icu_hour'] = np.arange(len(df))
        df['ventilation_status'] = (fio2 > 0.4).astype(int)
        
        if 'dialysis_status' not in df.columns:
            df['dialysis_status'] = 0.0
        if 'cumulative_fluid_24h' not in df.columns:
            df['cumulative_fluid_24h'] = 0.0

        # Add static features
        static_features = self._extract_static_features(icustay_id)
        for k, v in static_features.items():
            df[k] = v
            
        expected_features = (
            list(self.config.vitals) +
            list(self.config.labs) +
            list(self.config.derived) +
            list(self.config.static) +
            list(self.config.contextual)
        )

        # Add any missing columns with NaN
        for feat in expected_features:
            if feat not in df.columns:
                df[feat] = np.nan
                
        # Ensure column order is consistent
        df = df[expected_features].copy()
        
        # Final imputation
        df = df.fillna(0)

        if df.dropna(how='all').empty:
            raise ValueError(f"No valid features extracted for ICU stay {icustay_id}.")

        return df
    
    def _create_default_features(self, hours):
        """Create default feature dataframe when data is missing"""
        df = pd.DataFrame(index=hours)
        
        # Fill with reasonable defaults
        for v in self.config.vitals:
            df[v] = np.nan
        for l in self.config.labs:
            df[l] = np.nan
        for d in self.config.derived:
            df[d] = 0.0
        for s in self.config.static:
            defaults = {'age': 65.0, 'gender': 1.0, 'weight': 75.0}
            df[s] = defaults.get(s, 0.0)
        for c in self.config.contextual:
            if c == 'hour_of_day':
                df[c] = df.index.hour
            elif c == 'icu_hour':
                df[c] = np.arange(len(df))
            else:
                df[c] = 0.0
        
        return df

class FeatureImputer:

    def __init__(self):
        self.population_medians = None
        self.population_stds = None
        
    def fit(self, df_train: pd.DataFrame):
        """Learn population statistics from training data"""
        self.population_medians = df_train.median()
        self.population_stds = df_train.std()
        print(f"  Learned population statistics from {len(df_train)} training samples")
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df_imputed = df.copy()
        
        # Add missingness indicators BEFORE imputation
        for col in df.columns:
            df_imputed[f'{col}_missing'] = df[col].isna().astype(float)
        
        # Strategy 1: Vitals - forward-fill up to 2 hours
        vital_keywords = ['heart_rate', 'sbp', 'dbp', 'map', 'resp_rate', 'spo2', 
                         'temperature', 'gcs', 'shock_index', 'pf_ratio']
        vital_cols = [c for c in df.columns if any (v in c.lower() for v in vital_keywords)]
                      
        if vital_cols:
            df_imputed[vital_cols] = df_imputed[vital_cols].ffill(limit=2)
        
        # Strategy 2: Labs - population median
        if self.population_medians is not None:
            lab_cols = [c for c in df.columns if c in self.population_medians.index]
            for col in lab_cols:
                df_imputed[col] = df_imputed[col].fillna(self.population_medians[col])
        
        # Strategy 3: Any remaining NaNs - fill with 0
        df_imputed = df_imputed.fillna(0)
        
        return df_imputed

def fit_imputer_on_cohort(cohort_df: pd.DataFrame, feature_extractor: FeatureExtractor) -> FeatureImputer:

    print("\n🔧 Fitting FeatureImputer on raw cohort data...")
    
    imputer = FeatureImputer()
    all_raw_features = []
    
    for idx, patient in tqdm(cohort_df.iterrows(), total=len(cohort_df), 
                             desc="Collecting raw features for imputer"):
        try:
            df_features = feature_extractor.extract_hourly_features(
                patient['icustay_id'], 
                patient['intime'], 
                patient['outtime']
            )
            all_raw_features.append(df_features)
        except Exception:
            continue
    
    if all_raw_features:
        raw_cohort_data = pd.concat(all_raw_features, ignore_index=True)
        imputer.fit(raw_cohort_data)
        print(f"✅ Imputer fitted successfully on {len(raw_cohort_data):,} hourly records")
    else:
        print("⚠️  No data for fitting imputer.")
    
    return imputer

@dataclass
class ActionBins:
    """Configuration for discretizing fluid and vasopressor actions."""
    
    # Fluid bins (mL/hour)
    fluid_bins: np.ndarray = field(default_factory=lambda: np.array([0, 250, 500, 1000, 2000, np.inf]))
    fluid_labels: List[str] = field(default_factory=lambda: ["none", "low", "moderate", "high", "aggressive"])
    
    # Vasopressor bins (mcg/kg/min norepinephrine-equivalent)
    vaso_bins: np.ndarray = field(default_factory=lambda: np.array([0, 0.05, 0.15, 0.3, 0.6, np.inf]))
    vaso_labels: List[str] = field(default_factory=lambda: ["none", "low", "moderate", "high", "very_high"])
    
    # Total discrete actions: 5 × 5 = 25
    n_actions: int = 25


class ActionDiscretizer:

    def __init__(self, bins_config: ActionBins, data_dir: Optional[str] = None):
        self.config = bins_config
        
        # Resolve data_dir with fallback to global DATA_DIR
        resolved_path = data_dir if data_dir else DATA_DIR
        if not os.path.exists(os.path.join(resolved_path, 'INPUTEVENTS_MV.csv')):
            logger.warning(f"Data directory '{resolved_path}' missing INPUTEVENTS_MV.csv. Falling back to global DATA_DIR: {DATA_DIR}")
            resolved_path = DATA_DIR
            
        self.data_dir = resolved_path
        
        # Fluid itemids (crystalloids)
        self.FLUID_IDS = [225158, 225828, 225168, 226368]
        
        # Vasopressor itemids (Metavision)
        self.VASO_IDS: Dict[str, List[int]] = {
            'norepinephrine': [221906],
            'epinephrine': [221289],
            'dopamine': [221662],
            'vasopressin': [222315],
            'phenylephrine': [221749],
        }
        
        # Pre-load INPUTEVENTS_MV once
        self._inputevents: Dict[int, pd.DataFrame] = self._load_inputevents()
        logger.info(f"ActionDiscretizer initialized with data for {len(self._inputevents)} icustays.")

    def _load_inputevents(self) -> Dict[int, pd.DataFrame]:
        """Load INPUTEVENTS_MV and group by icustay_id."""
        path = os.path.join(self.data_dir, 'INPUTEVENTS_MV.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f"INPUTEVENTS_MV.csv not found at: {path}")
        
        usecols = ['icustay_id', 'itemid', 'starttime', 'endtime', 'rate', 
                   'rateuom', 'amount', 'amountuom']
        df = pd.read_csv(
            path, 
            usecols=usecols, 
            dtype={'icustay_id': 'int32', 'itemid': 'int32'}
        )
        
        # Group by icustay_id for fast access
        grouped = dict(iter(df.groupby('icustay_id')))
        return grouped

    def extract_actions(
        self,
        icustay_id: int,
        timestamps: pd.DatetimeIndex,
        weight_kg: float
    ) -> np.ndarray:

        # Get pre-loaded data for this patient
        inputevents = self._inputevents.get(icustay_id, pd.DataFrame())
        if inputevents.empty:
            logger.warning(f"No inputevents found for icustay_id {icustay_id}")
            return np.zeros((len(timestamps), 2), dtype=int)

        inputevents = inputevents.copy()
        inputevents['starttime'] = pd.to_datetime(inputevents['starttime'])
        inputevents['endtime'] = pd.to_datetime(inputevents['endtime'])

        start_global = timestamps[0]
        end_global = timestamps[-1] + pd.Timedelta(hours=1)

        inputevents = inputevents[
            (inputevents['endtime'] >= start_global) &
            (inputevents['starttime'] <= end_global)
        ].copy()

        if inputevents.empty:
            return np.zeros((len(timestamps), 2), dtype=int)

        # Build master time grid for accurate integration
        time_points = np.unique(
            np.concatenate([
                timestamps.values,
                (timestamps + pd.Timedelta(hours=1)).values,
                inputevents['starttime'].values,
                inputevents['endtime'].values
            ])
        )
        time_points = np.sort(time_points)
        t_seconds = time_points.astype('datetime64[s]').astype(np.int64)

        def time_to_idx(series: pd.Series) -> np.ndarray:
            return np.searchsorted(time_points, series.values)

        # ====================== FLUIDS ======================
        fluid_events = inputevents[inputevents['itemid'].isin(self.FLUID_IDS)].copy()
        if not fluid_events.empty:
            rateuom = fluid_events['rateuom'].astype(str).str.lower()
            amountuom = fluid_events['amountuom'].astype(str).str.lower()
            has_rate = fluid_events['rate'].notna().values
            has_amount = fluid_events['amount'].notna().values

            rate = np.zeros(len(fluid_events))

            # Rate-based fluids
            rate[has_rate & rateuom.str.contains('ml/hr')] = fluid_events.loc[
                has_rate & rateuom.str.contains('ml/hr'), 'rate'
            ]
            rate[has_rate & rateuom.str.contains('ml/min')] = fluid_events.loc[
                has_rate & rateuom.str.contains('ml/min'), 'rate'
            ] * 60
            rate[has_rate & rateuom.str.contains('l/hr')] = fluid_events.loc[
                has_rate & rateuom.str.contains('l/hr'), 'rate'
            ] * 1000

            # Amount-based (bolus → average rate)
            amount_ml = np.zeros(len(fluid_events))
            amount_ml[amountuom.str.contains('ml')] = fluid_events.loc[
                amountuom.str.contains('ml'), 'amount'
            ]
            amount_ml[amountuom.str.contains('l')] = fluid_events.loc[
                amountuom.str.contains('l'), 'amount'
            ] * 1000

            duration_hr = (
                (fluid_events['endtime'] - fluid_events['starttime'])
                .dt.total_seconds() / 3600
            ).values

            valid_amount = has_amount & (duration_hr > 0)
            rate[valid_amount] = amount_ml[valid_amount] / duration_hr[valid_amount]

            # Convert to mL/sec
            rate = rate / 3600

            # Piecewise integration
            start_idx = time_to_idx(fluid_events['starttime'])
            end_idx = time_to_idx(fluid_events['endtime'])

            diff = np.zeros(len(time_points))
            np.add.at(diff, start_idx, rate)
            np.add.at(diff, end_idx, -rate)

            rate_signal = np.cumsum(diff)
            dt = np.diff(t_seconds, prepend=t_seconds[0])
            fluid_cum = np.cumsum(rate_signal * dt)

            # Hourly fluid volume
            start_idx = time_to_idx(pd.Series(timestamps))
            end_idx = time_to_idx(pd.Series(timestamps + pd.Timedelta(hours=1)))
            
            start_idx = np.clip(start_idx, 0, len(time_points) - 1)
            end_idx = np.clip(end_idx, 0, len(time_points) - 1)
            
            fluid_hourly = fluid_cum[end_idx] - fluid_cum[start_idx]
            fluid_hourly = np.atleast_1d(fluid_hourly)
        
        else:
            fluid_hourly = np.zeros(len(timestamps))

        # ====================== VASOPRESSORS ======================
        vaso_sum_diff = np.zeros(len(time_points))
        vaso_time_diff = np.zeros(len(time_points))

        for drug, ids in self.VASO_IDS.items():
            sub = inputevents[inputevents['itemid'].isin(ids)].copy()
            if sub.empty:
                continue

            rateuom = sub['rateuom'].astype(str).str.lower()
            rate = sub['rate'].values
            rate_std = np.zeros(len(sub))

            # Unit normalization
            mask = rateuom.str.contains('mcg/kg/min')
            rate_std[mask] = rate[mask]

            mask = rateuom.str.contains('mcg/kg/hr')
            rate_std[mask] = rate[mask] / 60

            mask = rateuom.str.contains('mcg/min')
            rate_std[mask] = rate[mask] / weight_kg

            # Vasopressin units handling
            mask = rateuom.str.contains('units/min') & (drug == 'vasopressin')
            rate_std[mask] = rate[mask]
            mask = rateuom.str.contains('units/hr') & (drug == 'vasopressin')
            rate_std[mask] = rate[mask] / 60

            # Norepinephrine equivalence
            if drug == 'dopamine':
                rate_std /= 100
            elif drug == 'phenylephrine':
                rate_std /= 10
            elif drug == 'vasopressin':
                rate_std *= 2.5

            # Integration
            start_idx = time_to_idx(sub['starttime'])
            end_idx = time_to_idx(sub['endtime'])

            np.add.at(vaso_sum_diff, start_idx, rate_std)
            np.add.at(vaso_sum_diff, end_idx, -rate_std)
            np.add.at(vaso_time_diff, start_idx, 1)
            np.add.at(vaso_time_diff, end_idx, -1)

        vaso_rate_signal = np.cumsum(vaso_sum_diff)
        vaso_mask = np.cumsum(vaso_time_diff) > 0

        dt = np.diff(t_seconds, prepend=t_seconds[0])
        vaso_num = np.cumsum(vaso_rate_signal * dt)
        vaso_den = np.cumsum(vaso_mask.astype(float) * dt)

        # === HOURLY AVERAGE VASO RATE ===
        # Compute indices on the same time grid as fluids
        start_idx = time_to_idx(pd.Series(timestamps))
        end_idx = time_to_idx(pd.Series(timestamps + pd.Timedelta(hours=1)))
        
        start_idx = np.clip(start_idx, 0, len(time_points) - 1)
        end_idx = np.clip(end_idx, 0, len(time_points) - 1)
        
        num = vaso_num[end_idx] - vaso_num[start_idx]
        den = vaso_den[end_idx] - vaso_den[start_idx]
        vaso_hourly = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        vaso_hourly = np.atleast_1d(vaso_hourly)

        # ====================== DISCRETIZATION ======================
        fluid_bins = np.digitize(fluid_hourly, self.config.fluid_bins[1:-1])
        vaso_bins = np.digitize(vaso_hourly, self.config.vaso_bins[1:-1])

        try:
            return np.stack([fluid_bins, vaso_bins], axis=1)
        except ValueError:
            # Fallback if shapes don't match
            return np.zeros((len(timestamps), 2), dtype=int)

    def action_to_index(self, fluid_bin: int, vaso_bin: int) -> int:
        """Convert (fluid_bin, vaso_bin) → single action index 0-24"""
        return fluid_bin * 5 + vaso_bin

    def index_to_action(self, action_idx: int) -> Tuple[int, int]:
        """Convert single action index → (fluid_bin, vaso_bin)"""
        fluid_bin = action_idx // 5
        vaso_bin = action_idx % 5
        return fluid_bin, vaso_bin

@dataclass
class Trajectory:

    icustay_id: int
    states: np.ndarray  # [T, D_state]
    actions: np.ndarray  # [T, 2] or [T] (action indices)
    length: int
    mortality_48h: bool
    mortality_hosp: bool
    initial_sofa: float
    max_sofa: float
    age: float
    gender: str
    weight: float
    intime: pd.Timestamp
    
    # Will be added later during reward learning
    rewards: Optional[np.ndarray] = None
    returns_to_go: Optional[np.ndarray] = None
 
 
class TrajectoryBuilder:

    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 action_discretizer: ActionDiscretizer,
                 feature_imputer: FeatureImputer,
                 min_length: int = 24,
                 max_length: int = 168):
        self.feature_extractor = feature_extractor
        self.action_discretizer = action_discretizer
        self.feature_imputer = feature_imputer
        self.min_length = min_length
        self.max_length = max_length
        
    def build_trajectory(self, patient_row: pd.Series) -> Optional[Trajectory]:

        icustay_id = patient_row['icustay_id']
        intime = patient_row['intime']
        outtime = patient_row['outtime']
        weight = patient_row.get('weight', 75.0)
        
        # Extract features
        df_features = self.feature_extractor.extract_hourly_features(
            icustay_id, intime, outtime
        )
        
        # Extract actions
        actions = self.action_discretizer.extract_actions(
            icustay_id, df_features.index, weight
        )
        
        # Quality check 1: minimum length
        if len(df_features) < self.min_length:
            return None
        
        # Truncate to max length
        if len(df_features) > self.max_length:
            df_features = df_features.iloc[:self.max_length]
            actions = actions[:self.max_length]
        
        # Quality check 2: missing data threshold
        missing_ratio = df_features.isna().mean().mean()
        if missing_ratio > 0.5:  # >50% missing
            return None
        
        # Impute missing values
        df_imputed = self.feature_imputer.transform(df_features)
        
        if 'sofa_score' in df_features.columns and len(df_features) > 0:
            sofa_series = pd.to_numeric(df_features['sofa_score'], errors='coerce').fillna(0)
            initial_sofa = float(sofa_series.iloc[0])
            max_sofa = float(sofa_series.max())
        else:
            initial_sofa = 0.0
            max_sofa = 0.0
                    
        # Convert to numpy
        states = df_imputed.values.astype(np.float32)
        
        # Create trajectory object
        trajectory = Trajectory(
            icustay_id=int(icustay_id),
            states=states,
            actions=actions,
            length=len(states),
            mortality_48h=bool(patient_row.get('mortality_48h', 0)),
            mortality_hosp=bool(patient_row.get('mortality_hosp', 0)),
            initial_sofa=initial_sofa,      
            max_sofa=max_sofa,             
            age=float(patient_row.get('age', 65)),
            gender=str(patient_row.get('gender', 'M')),
            weight=float(weight),
            intime=intime
        )
        
        return trajectory
    
    def build_dataset(self, cohort_df: pd.DataFrame) -> List[Trajectory]:
        """Build all trajectories from cohort"""
        trajectories = []
        
        print("🔧 Building trajectories...")
        for idx, patient in tqdm(cohort_df.iterrows(), total=len(cohort_df), desc="Processing patients"):
            traj = self.build_trajectory(patient)
            if traj is not None:
                trajectories.append(traj)
        
        print("\n🔍 Trajectory Debug Info:")
        
        for traj in trajectories[:3]:
            print(f"\nICU Stay {traj.icustay_id}:")
            print(f"  State dim: {traj.states.shape[1]}")
            print(f"  SOFA (initial/max): {traj.initial_sofa}/{traj.max_sofa}")
            print(f"  Mortality: {traj.mortality_48h}")
            print(f"  Non-NaN features: {np.isfinite(traj.states).sum(axis=0).sum()} / {traj.states.size}")
        
        print(f"✅ Built {len(trajectories)} valid trajectories")
        print(f"⚠️  Failed {len(cohort_df) - len(trajectories)} quality checks")
        
        return trajectories

 
class DatasetSplitter:

    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
    def split(self, trajectories: List[Trajectory]) -> Dict[str, List[Trajectory]]:

        # Sort by admission time
        sorted_trajs = sorted(trajectories, key=lambda t: t.intime)
        
        n = len(sorted_trajs)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        
        splits = {
            'train': sorted_trajs[:n_train],
            'val': sorted_trajs[n_train:n_train+n_val],
            'test': sorted_trajs[n_train+n_val:]
        }
        
        # Log statistics
        print("\n📊 Dataset Split Statistics:")
        for split_name, split_data in splits.items():
            mortality_rate = np.mean([t.mortality_48h for t in split_data])
            mean_length = np.mean([t.length for t in split_data])
            mean_sofa = np.mean([t.initial_sofa for t in split_data])
            
            print(f"  {split_name.upper():5s}: {len(split_data):4d} trajectories | "
                  f"Mortality: {mortality_rate:.1%} | "
                  f"Avg Length: {mean_length:.1f}h | "
                  f"Avg SOFA: {mean_sofa:.1f}")
        
        return splits

class FeatureNormalizer:

    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, train_trajectories: List[Trajectory]):
        """Compute statistics from training data only"""
        # Check and standardize dimensions
        dims = [t.states.shape[1] for t in train_trajectories]
        if len(set(dims)) > 1:
            print(f"⚠️  Warning: Inconsistent state dimensions: {set(dims)}")
            print(f"   Using most common dimension")
            
            # Find most common dimension
            most_common_dim = Counter(dims).most_common(1)[0][0]
            
            # Filter to only trajectories with that dimension
            train_trajectories = [t for t in train_trajectories if t.states.shape[1] == most_common_dim]
            print(f"   Kept {len(train_trajectories)} trajectories with dim={most_common_dim}")
        
        all_states = np.concatenate([t.states for t in train_trajectories], axis=0)
        
        self.mean = all_states.mean(axis=0)
        self.std = all_states.std(axis=0) + 1e-6
        
        print(f"\n📐 Feature Normalization Statistics:")
        print(f"  State dimension: {len(self.mean)}")
        print(f"  Mean range: [{self.mean.min():.2f}, {self.mean.max():.2f}]")
        print(f"  Std range: [{self.std.min():.2f}, {self.std.max():.2f}]")
       
    # Apply Z-score normalization 
    def transform(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        if self.mean is None or self.std is None:
            raise ValueError("Must call fit() before transform()")
        
        normalized = []
        expected_dim = len(self.mean)
        
        for traj in trajectories:
            # Skip trajectories with wrong dimension
            if traj.states.shape[1] != expected_dim:
                print(f"Skipping trajectory {traj.icustay_id}: dim={traj.states.shape[1]}, expected={expected_dim}")
                continue
            traj_copy = deepcopy(traj)
            traj_copy.states = (traj.states - self.mean) / self.std
            normalized.append(traj_copy)
        
        return normalized
    
    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std)
        print(f"💾 Saved normalization parameters to {path}")
    
    def load(self, path: str):
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']
        print(f"📂 Loaded normalization parameters from {path}")