"""
Unit tests for the DataLoader component.
Tests file reading, data alignment, and timestamp handling.
"""

import pandas as pd
import pytest
from pathlib import Path
import gzip
import json
from datetime import datetime, timedelta

from src.data_ingestion.loader import DataLoader


class TestDataLoader:
    """Test suite for DataLoader component."""
    
    @pytest.fixture
    def sample_l2_data(self) -> pd.DataFrame:
        """Create sample L2 order book data for testing."""
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        timestamps = [base_time + timedelta(seconds=i * 5) for i in range(5)]
        
        data = {
            'timestamp': [ts.isoformat() for ts in timestamps],
        }
        
        # Add 5 levels of bid/ask prices and volumes
        for level in range(1, 6):
            data[f'bid_price_{level}'] = [2000.0 - level * 0.1 + i * 0.1 for i in range(5)]
            data[f'ask_price_{level}'] = [2000.1 + level * 0.1 + i * 0.1 for i in range(5)]
            data[f'bid_volume_{level}'] = [1000 + level * 100 + i * 100 for i in range(5)]
            data[f'ask_volume_{level}'] = [1100 + level * 100 + i * 100 for i in range(5)]
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_l2_data(self) -> pd.DataFrame:
        """Create sample L2 order book data for testing."""
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        # L2 data every 5 seconds (less frequent than tick data)
        timestamps = [base_time + timedelta(seconds=i * 5) for i in range(5)]
        
        return pd.DataFrame({
            'timestamp': [ts.isoformat() for ts in timestamps],
            'bid_depth_1': [2000.0 + i * 0.1 for i in range(5)],
            'ask_depth_1': [2000.1 + i * 0.1 for i in range(5)],
            'bid_volume_1': [1000 + i * 100 for i in range(5)],
            'ask_volume_1': [1100 + i * 100 for i in range(5)]
        })
    
    def create_test_files(self, tmp_path: Path, tick_data: pd.DataFrame, l2_data: pd.DataFrame) -> tuple[Path, Path]:
        """Create temporary gzipped CSV files for testing."""
        tick_file = tmp_path / "tick_data.csv.gz"
        l2_file = tmp_path / "l2_data.csv.gz"
        
        # Ensure bid_volume column exists in tick_data
        if 'bid_volume' not in tick_data.columns:
            tick_data = tick_data.copy()
            tick_data['bid_volume'] = tick_data.get('volume', [100] * len(tick_data))
        
        # Write tick data as gzipped CSV
        with gzip.open(tick_file, 'wt', encoding='utf-8') as f:
            tick_data.to_csv(f, index=False)
        
        # Write L2 data as gzipped CSV
        with gzip.open(l2_file, 'wt', encoding='utf-8') as f:
            l2_data.to_csv(f, index=False)
        
        return tick_file, l2_file
    
    def test_data_loader_initialization(self):
        """Test DataLoader can be instantiated."""
        loader = DataLoader()
        assert loader is not None
    
    def test_load_and_align_success(self, tmp_path: Path, sample_tick_data: pd.DataFrame, sample_l2_data: pd.DataFrame):
        """Test successful loading and alignment of tick and L2 data."""
        loader = DataLoader()
        tick_file, l2_file = self.create_test_files(tmp_path, sample_tick_data, sample_l2_data)
        
        result_df = loader.load_and_align(str(tick_file), str(l2_file))
        
        # Verify basic structure
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(sample_tick_data)  # Should have same length as tick data
        
        # Verify columns from both datasets are present
        assert 'bid' in result_df.columns
        assert 'ask' in result_df.columns
        assert 'bid_depth_1' in result_df.columns
        assert 'ask_depth_1' in result_df.columns
        
        # Verify timestamp alignment worked
        assert pd.api.types.is_datetime64_any_dtype(result_df.index)
        
        # Verify no NaN values in critical columns after alignment
        assert not result_df['bid'].isna().any()
        assert not result_df['ask'].isna().any()
    
    def test_load_and_align_time_alignment(self, tmp_path: Path, sample_tick_data: pd.DataFrame, sample_l2_data: pd.DataFrame):
        """Test that merge_asof correctly aligns timestamps backward."""
        loader = DataLoader()
        tick_file, l2_file = self.create_test_files(tmp_path, sample_tick_data, sample_l2_data)
        
        result_df = loader.load_and_align(str(tick_file), str(l2_file))
        
        # First tick should align with first L2 record
        assert result_df.iloc[0]['bid_depth_1'] == sample_l2_data.iloc[0]['bid_depth_1']
        
        # Tick at 3 seconds should still use L2 from 0 seconds (backward fill)
        tick_3s_idx = 3
        assert result_df.iloc[tick_3s_idx]['bid_depth_1'] == sample_l2_data.iloc[0]['bid_depth_1']
        
        # Tick at 7 seconds should use L2 from 5 seconds
        tick_7s_idx = 7
        assert result_df.iloc[tick_7s_idx]['bid_depth_1'] == sample_l2_data.iloc[1]['bid_depth_1']
    
    def test_load_empty_files(self, tmp_path: Path):
        """Test handling of empty CSV files."""
        loader = DataLoader()
        
        # Create empty files
        tick_file = tmp_path / "empty_tick.csv.gz"
        l2_file = tmp_path / "empty_l2.csv.gz"
        
        empty_df = pd.DataFrame(columns=['timestamp', 'bid', 'ask', 'volume'])
        empty_l2_df = pd.DataFrame(columns=['timestamp', 'bid_depth_1', 'ask_depth_1', 'bid_volume_1', 'ask_volume_1'])
        
        with gzip.open(tick_file, 'wt', encoding='utf-8') as f:
            empty_df.to_csv(f, index=False)
        
        with gzip.open(l2_file, 'wt', encoding='utf-8') as f:
            empty_l2_df.to_csv(f, index=False)
        
        result_df = loader.load_and_align(str(tick_file), str(l2_file))
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0
    
    def test_load_nonexistent_files(self):
        """Test error handling for nonexistent files."""
        loader = DataLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_and_align("nonexistent_tick.csv.gz", "nonexistent_l2.csv.gz")
    
    def test_load_malformed_csv(self, tmp_path: Path):
        """Test handling of malformed CSV files."""
        loader = DataLoader()
        
        # Create malformed CSV file
        tick_file = tmp_path / "malformed_tick.csv.gz"
        
        with gzip.open(tick_file, 'wt', encoding='utf-8') as f:
            f.write("invalid,csv,data\nwithout,proper,headers\n")
        
        l2_file = tmp_path / "valid_l2.csv.gz"
        sample_l2 = pd.DataFrame({
            'timestamp': ['2024-01-01T09:00:00'],
            'bid_depth_1': [2000.0],
            'ask_depth_1': [2000.1],
            'bid_volume_1': [1000],
            'ask_volume_1': [1100]
        })
        
        with gzip.open(l2_file, 'wt', encoding='utf-8') as f:
            sample_l2.to_csv(f, index=False)
        
        # Should raise an exception due to missing expected columns
        with pytest.raises((KeyError, ValueError)):
            loader.load_and_align(str(tick_file), str(l2_file))
    
    def test_timestamp_parsing(self, tmp_path: Path):
        """Test various timestamp formats are handled correctly."""
        loader = DataLoader()
        
        # Create data with different timestamp formats
        tick_data = pd.DataFrame({
            'timestamp': [
                '2024-01-01T09:00:00.000Z',
                '2024-01-01T09:00:01.500Z',
                '2024-01-01T09:00:02.999Z'
            ],
            'bid': [2000.0, 2000.1, 2000.2],
            'ask': [2000.1, 2000.2, 2000.3],
            'volume': [100, 110, 120]
        })
        
        l2_data = pd.DataFrame({
            'timestamp': ['2024-01-01T09:00:00.000Z'],
            'bid_depth_1': [2000.0],
            'ask_depth_1': [2000.1],
            'bid_volume_1': [1000],
            'ask_volume_1': [1100]
        })
        
        tick_file, l2_file = self.create_test_files(tmp_path, tick_data, l2_data)
        
        result_df = loader.load_and_align(str(tick_file), str(l2_file))
        
        # Verify timestamps were parsed correctly
        assert pd.api.types.is_datetime64_any_dtype(result_df.index)
        assert len(result_df) == 3
        
        # Verify microsecond precision is preserved
        time_diffs = result_df.index.to_series().diff().dropna()
        assert any(diff.total_seconds() < 1.0 for diff in time_diffs)
    
    def test_large_dataset_performance(self, tmp_path: Path):
        """Test performance with larger datasets."""
        loader = DataLoader()
        
        # Create larger dataset (1000 records)
        base_time = datetime(2024, 1, 1, 9, 0, 0)
        tick_timestamps = [base_time + timedelta(milliseconds=i * 100) for i in range(1000)]
        l2_timestamps = [base_time + timedelta(seconds=i) for i in range(100)]
        
        large_tick_data = pd.DataFrame({
            'timestamp': [ts.isoformat() for ts in tick_timestamps],
            'bid': [2000.0 + (i % 100) * 0.01 for i in range(1000)],
            'ask': [2000.1 + (i % 100) * 0.01 for i in range(1000)],
            'volume': [100 + i for i in range(1000)]
        })
        
        large_l2_data = pd.DataFrame({
            'timestamp': [ts.isoformat() for ts in l2_timestamps],
            'bid_depth_1': [2000.0 + i * 0.1 for i in range(100)],
            'ask_depth_1': [2000.1 + i * 0.1 for i in range(100)],
            'bid_volume_1': [1000 + i * 10 for i in range(100)],
            'ask_volume_1': [1100 + i * 10 for i in range(100)]
        })
        
        tick_file, l2_file = self.create_test_files(tmp_path, large_tick_data, large_l2_data)
        
        # This should complete without timeout or memory issues
        result_df = loader.load_and_align(str(tick_file), str(l2_file))
        
        assert len(result_df) == 1000
        assert not result_df.isna().all().any()  # No completely empty columns