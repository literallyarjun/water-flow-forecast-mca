import os
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from datetime import datetime

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ['date', 'consumption']
OPTIONAL_COLUMNS = ['temperature', 'humidity', 'rainfall', 'wind', 'evaporation']
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"]


@dataclass
class DataProvenance:
    source: str
    filename: Optional[str]
    original_rows: int
    filtered_rows: int
    year: int
    validation_errors: List[str]
    load_timestamp: datetime
    
    @property
    def is_valid(self) -> bool:
        return len(self.validation_errors) == 0 and self.filtered_rows > 0
    
    @property
    def source_description(self) -> str:
        if self.source == 'uploaded':
            return f"Using uploaded data: {self.filename}"
        elif self.source == 'default':
            return f"Using 2025 dataset: {self.filename}"
        else:
            return "Using generated 2025 sample data"


class DataValidator:
    @staticmethod
    def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors = []
        
        if df.empty:
            errors.append("Dataset is empty")
            return False, errors
        
        has_date = 'date' in df.columns or 'Date' in df.columns
        if not has_date:
            errors.append("Missing required column: date/Date")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            errors.append("No numeric columns found for consumption data")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> List[str]:
        warnings = []
        
        if 'consumption' in df.columns:
            null_count = df['consumption'].isnull().sum()
            if null_count > 0:
                warnings.append(f"Found {null_count} missing consumption values")
            
            negative_count = (df['consumption'] < 0).sum()
            if negative_count > 0:
                warnings.append(f"Found {negative_count} negative consumption values")
        
        return warnings


def parse_date_column(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    df = df.copy()
    
    for fmt in DATE_FORMATS:
        try:
            df[date_col] = pd.to_datetime(df[date_col], format=fmt)
            logger.info(f"Successfully parsed dates using format: {fmt}")
            return df
        except (ValueError, TypeError):
            continue
    
    try:
        df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=False)
        logger.info("Parsed dates using flexible parsing")
        return df
    except Exception:
        pass
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    invalid_count = df[date_col].isnull().sum()
    if invalid_count > 0:
        logger.warning(f"Could not parse {invalid_count} date values, they will be dropped")
        df = df.dropna(subset=[date_col])
    
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    column_mapping = {
        'Date': 'date',
        'DATE': 'date',
        'Consumption': 'consumption',
        'CONSUMPTION': 'consumption',
        'Water_Consumption': 'consumption',
        'water_consumption': 'consumption',
        'Usage': 'consumption',
        'Temperature': 'temperature',
        'Humidity': 'humidity',
        'Rainfall': 'rainfall',
        'Wind': 'wind',
        'Evaporation': 'evaporation'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    if 'date' not in df.columns and len(df.columns) > 0:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: 'date'})
        logger.info(f"Renamed first column '{first_col}' to 'date'")
    
    if 'consumption' not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df = df.rename(columns={numeric_cols[0]: 'consumption'})
            logger.info(f"Renamed '{numeric_cols[0]}' to 'consumption'")
    
    return df


class DataIngestionService:
    def __init__(self, upload_folder: str = 'uploads', default_dataset: str = 'metroflow_2025_full_year.csv', target_year: int = 2025):
        self.upload_folder = upload_folder
        self.default_dataset = default_dataset
        self.target_year = target_year
        self.validator = DataValidator()
        
        os.makedirs(upload_folder, exist_ok=True)
    
    def load_uploaded_file(self, filepath: str) -> Tuple[Optional[pd.DataFrame], DataProvenance]:
        filename = os.path.basename(filepath)
        
        try:
            df = pd.read_csv(filepath)
            original_rows = len(df)
            
            is_valid, errors = self.validator.validate_schema(df)
            if not is_valid:
                return None, DataProvenance(
                    source='uploaded',
                    filename=filename,
                    original_rows=original_rows,
                    filtered_rows=0,
                    year=self.target_year,
                    validation_errors=errors,
                    load_timestamp=datetime.now()
                )
            
            df = normalize_columns(df)
            df = parse_date_column(df)
            df = df[df['date'].dt.year == self.target_year]
            
            quality_warnings = self.validator.validate_data_quality(df)
            for warning in quality_warnings:
                logger.warning(warning)
            
            provenance = DataProvenance(
                source='uploaded',
                filename=filename,
                original_rows=original_rows,
                filtered_rows=len(df),
                year=self.target_year,
                validation_errors=[],
                load_timestamp=datetime.now()
            )
            
            logger.info(f"Loaded {len(df)} rows from {filename} (filtered to {self.target_year})")
            return df, provenance
            
        except Exception as e:
            logger.error(f"Error loading file {filename}: {str(e)}")
            return None, DataProvenance(
                source='uploaded',
                filename=filename,
                original_rows=0,
                filtered_rows=0,
                year=self.target_year,
                validation_errors=[str(e)],
                load_timestamp=datetime.now()
            )
    
    def load_default_dataset(self) -> Tuple[pd.DataFrame, DataProvenance]:
        default_path = os.path.join(self.upload_folder, self.default_dataset)
        
        if os.path.exists(default_path):
            df, provenance = self.load_uploaded_file(default_path)
            if df is not None and not df.empty:
                provenance.source = 'default'
                return df, provenance
        
        logger.info(f"Default dataset not found, generating {self.target_year} sample data")
        df = self._generate_sample_data()
        
        provenance = DataProvenance(
            source='generated',
            filename=None,
            original_rows=len(df),
            filtered_rows=len(df),
            year=self.target_year,
            validation_errors=[],
            load_timestamp=datetime.now()
        )
        
        return df, provenance
    
    def _generate_sample_data(self, days: int = 365) -> pd.DataFrame:
        np.random.seed(42)
        dates = pd.date_range(start=f'{self.target_year}-01-01', periods=days, freq='D')
        
        base_consumption = 100
        seasonal = 20 * np.sin(2 * np.pi * np.arange(days) / 365)
        weekly = 10 * np.sin(2 * np.pi * np.arange(days) / 7)
        noise = np.random.normal(0, 5, days)
        consumption = base_consumption + seasonal + weekly + noise
        
        temperature = 20 + 15 * np.sin(2 * np.pi * np.arange(days) / 365) + np.random.normal(0, 3, days)
        humidity = 60 + 20 * np.cos(2 * np.pi * np.arange(days) / 365) + np.random.normal(0, 5, days)
        rainfall = np.maximum(0, np.random.exponential(5, days) * (1 + 0.5 * np.sin(2 * np.pi * np.arange(days) / 365)))
        wind = 10 + 5 * np.random.random(days)
        evaporation = 3 + 2 * np.sin(2 * np.pi * np.arange(days) / 365) + np.random.normal(0, 0.5, days)
        
        return pd.DataFrame({
            'date': dates,
            'consumption': np.maximum(consumption, 20),
            'temperature': temperature,
            'humidity': np.clip(humidity, 20, 100),
            'rainfall': rainfall,
            'wind': wind,
            'evaporation': np.maximum(evaporation, 0)
        })
    
    def process_uploaded_files(self, filepaths: List[str]) -> Tuple[Optional[pd.DataFrame], DataProvenance]:
        all_data = []
        all_filenames = []
        total_original_rows = 0
        all_errors = []
        
        for filepath in filepaths:
            df, provenance = self.load_uploaded_file(filepath)
            if df is not None and not df.empty:
                all_data.append(df)
                all_filenames.append(provenance.filename)
                total_original_rows += provenance.original_rows
            else:
                all_errors.extend(provenance.validation_errors)
        
        if not all_data:
            return None, DataProvenance(
                source='uploaded',
                filename=', '.join(all_filenames) if all_filenames else None,
                original_rows=total_original_rows,
                filtered_rows=0,
                year=self.target_year,
                validation_errors=all_errors if all_errors else ["No valid data found"],
                load_timestamp=datetime.now()
            )
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        
        provenance = DataProvenance(
            source='uploaded',
            filename=', '.join(all_filenames),
            original_rows=total_original_rows,
            filtered_rows=len(combined_df),
            year=self.target_year,
            validation_errors=[],
            load_timestamp=datetime.now()
        )
        
        logger.info(f"Combined {len(all_data)} files: {len(combined_df)} rows after deduplication")
        return combined_df, provenance
