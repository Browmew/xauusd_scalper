# src/data_ingestion/loader.py
"""
Data Ingestion Module for XAUUSD High-Frequency Trading System.

This module provides efficient loading and alignment of tick-level and Level 2 
order book data from compressed CSV files. It handles millisecond-precision 
timestamp alignment and optimizes memory usage for large datasets.

Classes:
    DataLoader: Main class for loading and aligning historical market data.
"""

import gzip
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame

from ..utils.helpers import get_config_value, get_project_root, load_config
from ..utils.logging import get_logger


class DataLoader:
    """
    High-performance data loader for tick and Level 2 order book data.
    
    This class handles the ingestion, parsing, and alignment of historical
    market data from compressed CSV files. It optimizes for memory efficiency
    and processing speed while maintaining millisecond-precision alignment.
    
    Attributes:
        config (Dict): Loaded configuration dictionary
        logger: Structured logger instance
        tick_data_path (str): Path to tick data files
        l2_data_path (str): Path to L2 order book data files
    """
    
    def __init__(self) -> None:
        """Initialize the DataLoader with configuration and logging."""
        self.config = load_config()
        self.logger = get_logger(__name__)
        
        # Get data paths from configuration
        self.tick_data_path = get_config_value("data.sources.tick_data")
        self.l2_data_path = get_config_value("data.sources.l2_data")
        
        # Data type optimizations for memory efficiency - only include guaranteed columns
        self.tick_dtypes = {
            'timestamp': 'object',  # Will convert to datetime64[ms]
            'bid': 'float32',
            'ask': 'float32',
            'volume': 'float32'
        }
        
        self.l2_dtypes = {
            'timestamp': 'object',  # Will convert to datetime64[ms]
            'bid_price_1': 'float32',
            'bid_volume_1': 'float32',
            'bid_price_2': 'float32',
            'bid_volume_2': 'float32',
            'bid_price_3': 'float32',
            'bid_volume_3': 'float32',
            'bid_price_4': 'float32',
            'bid_volume_4': 'float32',
            'bid_price_5': 'float32',
            'bid_volume_5': 'float32',
            'ask_price_1': 'float32',
            'ask_volume_1': 'float32',
            'ask_price_2': 'float32',
            'ask_volume_2': 'float32',
            'ask_price_3': 'float32',
            'ask_volume_3': 'float32',
            'ask_price_4': 'float32',
            'ask_volume_4': 'float32',
            'ask_price_5': 'float32',
            'ask_volume_5': 'float32'
        }
        
        self.logger.info("DataLoader initialized", 
                        tick_path=self.tick_data_path,
                        l2_path=self.l2_data_path)
    
    def _validate_file_path(self, file_path: str) -> Path:
        """
        Validate and resolve file path.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Resolved Path object
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.isabs(file_path):
            # Resolve relative paths from project root
            project_root = get_project_root()
            full_path = project_root / file_path
        else:
            full_path = Path(file_path)
            
        if not full_path.exists():
            raise FileNotFoundError(f"Data file not found: {full_path}")
            
        return full_path
    
    def _optimize_timestamp_column(self, df: DataFrame, 
                                   timestamp_col: str = 'timestamp') -> DataFrame:
        """
        Convert timestamp column to optimized datetime64[ms] format.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            
        Returns:
            DataFrame with optimized timestamp column
        """
        # Convert to datetime with millisecond precision
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], 
                                         format='mixed', 
                                         utc=True).dt.floor('ms')
        
        # Set as index for efficient alignment operations
        df.set_index(timestamp_col, inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def load_tick_data(self, file_path: str, 
                    chunksize: Optional[int] = None) -> DataFrame:
        """Load tick data from compressed CSV file."""
        validated_path = self._validate_file_path(file_path)
        
        self.logger.info("Loading tick data", 
                        file_path=str(validated_path),
                        chunksize=chunksize)
        
        try:
            # Determine if file is compressed
            if validated_path.suffix == '.gz':
                compression = 'gzip'
            else:
                compression = None
            
            # Load data with optimized settings
            if chunksize:
                chunks = []
                chunk_reader = pd.read_csv(
                    validated_path,
                    engine='pyarrow',
                    compression=compression,
                    chunksize=chunksize,
                    parse_dates=False  # We'll handle datetime conversion manually
                )
                
                for chunk_num, chunk in enumerate(chunk_reader):
                    self.logger.debug("Processing tick chunk", chunk_num=chunk_num)
                    # Only apply dtypes for columns that exist
                    for col, dtype in self.tick_dtypes.items():
                        if col in chunk.columns and col != 'timestamp':
                            chunk[col] = chunk[col].astype(dtype)
                    chunk = self._optimize_timestamp_column(chunk)
                    chunks.append(chunk)
                
                df = pd.concat(chunks, axis=0).sort_index()
            else:
                df = pd.read_csv(
                    validated_path,
                    engine='pyarrow',
                    compression=compression,
                    parse_dates=False
                )
                # Only apply dtypes for columns that exist
                for col, dtype in self.tick_dtypes.items():
                    if col in df.columns and col != 'timestamp':
                        df[col] = df[col].astype(dtype)
                df = self._optimize_timestamp_column(df)
            
            # Remove any duplicate timestamps (keep last)
            df = df[~df.index.duplicated(keep='last')]
            
            self.logger.info("Tick data loaded successfully", 
                        rows=len(df),
                        memory_mb=df.memory_usage(deep=True).sum() / 1024**2,
                        start_time=df.index.min(),
                        end_time=df.index.max())
            
            return df
            
        except Exception as e:
            self.logger.error("Failed to load tick data", 
                            file_path=str(validated_path),
                            error=str(e))
            raise
    
    def load_l2_data(self, file_path: str, 
                    chunksize: Optional[int] = None) -> DataFrame:
        """
        Load Level 2 order book data from compressed CSV file.
        
        This method efficiently loads L2 order book snapshots with optimized
        data types. The L2 data typically contains multiple price levels for
        both bid and ask sides.
        
        Args:
            file_path: Path to the L2 data file (can be .csv or .csv.gz)
            chunksize: Optional chunk size for processing large files
            
        Returns:
            DataFrame with L2 data indexed by timestamp
            
        Time Complexity: O(n log n) where n is number of rows (due to sorting)
        Space Complexity: O(n) for the DataFrame
        """
        validated_path = self._validate_file_path(file_path)
        
        self.logger.info("Loading L2 data", 
                        file_path=str(validated_path),
                        chunksize=chunksize)
        
        try:
            # Determine if file is compressed
            if validated_path.suffix == '.gz':
                compression = 'gzip'
            else:
                compression = None
            
            # Load data with optimized settings
            if chunksize:
                chunks = []
                chunk_reader = pd.read_csv(
                    validated_path,
                    engine='pyarrow',
                    compression=compression,
                    chunksize=chunksize,
                    parse_dates=False
                )
                
                for chunk_num, chunk in enumerate(chunk_reader):
                    self.logger.debug("Processing L2 chunk", chunk_num=chunk_num)
                    # Only apply dtypes for columns that exist
                    for col, dtype in self.l2_dtypes.items():
                        if col in chunk.columns and col != 'timestamp':
                            chunk[col] = chunk[col].astype(dtype)
                    chunk = self._optimize_timestamp_column(chunk)
                    chunks.append(chunk)
                
                df = pd.concat(chunks, axis=0).sort_index()
            else:
                df = pd.read_csv(
                    validated_path,
                    engine='pyarrow',
                    compression=compression,
                    parse_dates=False
                )
                # Only apply dtypes for columns that exist
                for col, dtype in self.l2_dtypes.items():
                    if col in df.columns and col != 'timestamp':
                        df[col] = df[col].astype(dtype)
                df = self._optimize_timestamp_column(df)
            
            # Remove any duplicate timestamps (keep last)
            df = df[~df.index.duplicated(keep='last')]
            
            self.logger.info("L2 data loaded successfully", 
                        rows=len(df),
                        memory_mb=df.memory_usage(deep=True).sum() / 1024**2,
                        start_time=df.index.min() if len(df) > 0 else None,
                        end_time=df.index.max() if len(df) > 0 else None)
            
            return df
            
        except Exception as e:
            self.logger.error("Failed to load L2 data", 
                            file_path=str(validated_path),
                            error=str(e))
            raise
    
    def align_data(self, tick_df: DataFrame, l2_df: DataFrame,
                   method: str = 'forward_fill') -> DataFrame:
        """
        Align tick and L2 data with millisecond precision.
        
        This method performs temporal alignment of tick and L2 data, handling
        cases where timestamps don't exactly match. It uses forward-fill logic
        to propagate L2 snapshots until new data arrives.
        
        Args:
            tick_df: DataFrame with tick data (indexed by timestamp)
            l2_df: DataFrame with L2 data (indexed by timestamp)
            method: Alignment method ('forward_fill', 'nearest', 'exact')
            
        Returns:
            Aligned DataFrame with both tick and L2 data
            
        Time Complexity: O(n + m) where n, m are lengths of input DataFrames
        Space Complexity: O(n + m) for the result DataFrame
        """
        self.logger.info("Aligning tick and L2 data", 
                        tick_rows=len(tick_df),
                        l2_rows=len(l2_df),
                        method=method)
        
        if tick_df.empty or l2_df.empty:
            raise ValueError("Cannot align empty DataFrames")
        
        try:
            if method == 'forward_fill':
                # Use pandas' powerful reindex with forward fill
                # This will align L2 data to tick timestamps and forward-fill missing values
                l2_aligned = l2_df.reindex(tick_df.index, method='ffill')
                
                # Combine tick and L2 data
                aligned_df = pd.concat([tick_df, l2_aligned], axis=1)
                
            elif method == 'nearest':
                # Use nearest neighbor alignment
                l2_aligned = l2_df.reindex(tick_df.index, method='nearest')
                aligned_df = pd.concat([tick_df, l2_aligned], axis=1)
                
            elif method == 'exact':
                # Only keep rows where timestamps exist in both datasets
                common_index = tick_df.index.intersection(l2_df.index)
                aligned_df = pd.concat([
                    tick_df.loc[common_index],
                    l2_df.loc[common_index]
                ], axis=1)
                
            else:
                raise ValueError(f"Unknown alignment method: {method}")
            
            # Remove rows with all NaN values (if any)
            aligned_df.dropna(how='all', inplace=True)
            
            # Add alignment quality metrics
            alignment_ratio = len(aligned_df) / len(tick_df)
            
            self.logger.info("Data alignment completed", 
                           aligned_rows=len(aligned_df),
                           alignment_ratio=alignment_ratio,
                           start_time=aligned_df.index.min(),
                           end_time=aligned_df.index.max())
            
            if alignment_ratio < 0.5:
                self.logger.warning("Low alignment ratio detected", 
                                  ratio=alignment_ratio)
            
            return aligned_df
            
        except Exception as e:
            self.logger.error("Failed to align data", error=str(e))
            raise
    
    def load_and_align(self, tick_file: str, l2_file: str,
                   alignment_method: str = 'forward_fill',
                   chunksize: Optional[int] = None) -> DataFrame:
        """
        Main method to load both tick and L2 data and return aligned DataFrame.
        """
        self.logger.info("Starting data ingestion and alignment",
                        tick_file=tick_file,
                        l2_file=l2_file,
                        alignment_method=alignment_method)
        
        try:
            # Load tick data
            tick_df = self.load_tick_data(tick_file, chunksize=chunksize)
            
            # Load L2 data
            l2_df = self.load_l2_data(l2_file, chunksize=chunksize)
            
            # If both dataframes are empty, return empty result
            if tick_df.empty and l2_df.empty:
                return pd.DataFrame()
            
            # If one is empty, return the other
            if tick_df.empty:
                return l2_df
            if l2_df.empty:
                return tick_df
            
            # Align the data
            aligned_df = self.align_data(tick_df, l2_df, method=alignment_method)
            
            self.logger.info("Data ingestion completed successfully",
                        final_rows=len(aligned_df),
                        final_columns=len(aligned_df.columns),
                        memory_mb=aligned_df.memory_usage(deep=True).sum() / 1024**2)
            
            return aligned_df
            
        except Exception as e:
            self.logger.error("Data ingestion failed", error=str(e))
            raise
    
    def get_available_files(self, data_type: str = 'both') -> Dict[str, List[str]]:
        """
        Get list of available data files.
        
        Args:
            data_type: Type of data to search for ('tick', 'l2', 'both')
            
        Returns:
            Dictionary with available file paths
        """
        project_root = get_project_root()
        available_files = {}
        
        if data_type in ['tick', 'both']:
            tick_dir = project_root / self.tick_data_path
            if tick_dir.exists():
                available_files['tick'] = [
                    str(f) for f in tick_dir.glob('*.csv*')
                ]
            else:
                available_files['tick'] = []
        
        if data_type in ['l2', 'both']:
            l2_dir = project_root / self.l2_data_path
            if l2_dir.exists():
                available_files['l2'] = [
                    str(f) for f in l2_dir.glob('*.csv*')
                ]
            else:
                available_files['l2'] = []
        
        self.logger.info("Available files retrieved", 
                        data_type=data_type,
                        files=available_files)
        
        return available_files