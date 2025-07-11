#!/usr/bin/env python3
"""
Realistic L2 Order Book Data Generator for XAUUSD

This script generates synthetic but realistic Level 2 order book data
based on actual tick data, using market microstructure principles.
"""

import pandas as pd
import numpy as np
import gzip
from pathlib import Path
from datetime import datetime
import argparse


class L2DataGenerator:
    """Generates realistic L2 order book data from tick data."""
    
    def __init__(self, tick_size=0.01, levels=5):
        """
        Initialize the L2 generator.
        
        Args:
            tick_size: Minimum price increment for XAUUSD (0.01)
            levels: Number of price levels to generate (5)
        """
        self.tick_size = tick_size
        self.levels = levels
        
        # Market microstructure parameters for XAUUSD
        self.base_volume = 1000  # Base volume per level
        self.volume_decay = 0.7  # Volume decay factor per level
        self.spread_multiplier = 1.2  # How spread affects level spacing
        self.volatility_window = 20  # Ticks to calculate volatility
        self.imbalance_factor = 0.15  # Max order book imbalance
        
        # Random state for reproducible results
        self.rng = np.random.RandomState(42)
    
    def load_tick_data(self, tick_file_path):
        """Load tick data from compressed CSV file."""
        print(f"Loading tick data from {tick_file_path}")
        
        try:
            with gzip.open(tick_file_path, 'rt') as f:
                df = pd.read_csv(f)
            
            print(f"Found columns: {list(df.columns)}")
            print(f"Data shape: {df.shape}")
            
            # Auto-detect timestamp column
            timestamp_cols = [col for col in df.columns if any(word in col.lower() 
                             for word in ['time', 'date', 'timestamp'])]
            if timestamp_cols:
                timestamp_col = timestamp_cols[0]
                print(f"Using timestamp column: {timestamp_col}")
            else:
                # Use first column as timestamp
                timestamp_col = df.columns[0]
                print(f"No timestamp column found, using first column: {timestamp_col}")
            
            # Auto-detect bid/ask columns or create from available data
            bid_col = None
            ask_col = None
            
            # Look for explicit bid/ask columns
            for col in df.columns:
                if any(word in col.lower() for word in ['bid', 'buy']):
                    bid_col = col
                if any(word in col.lower() for word in ['ask', 'sell', 'offer']):
                    ask_col = col
            
            # If no bid/ask found, try to create from OHLC or single price
            if bid_col is None or ask_col is None:
                if 'close' in df.columns:
                    print("Creating bid/ask from close price with estimated spread")
                    df['bid'] = df['close'] - 0.05  # Estimate 5 pip spread
                    df['ask'] = df['close'] + 0.05
                    bid_col, ask_col = 'bid', 'ask'
                elif 'price' in df.columns:
                    print("Creating bid/ask from price with estimated spread")
                    df['bid'] = df['price'] - 0.05
                    df['ask'] = df['price'] + 0.05
                    bid_col, ask_col = 'bid', 'ask'
                else:
                    # Use first numeric column
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        price_col = numeric_cols[0]
                        print(f"Creating bid/ask from {price_col} with estimated spread")
                        df['bid'] = df[price_col] - 0.05
                        df['ask'] = df[price_col] + 0.05
                        bid_col, ask_col = 'bid', 'ask'
                    else:
                        raise ValueError("No suitable price columns found")
            
            print(f"Using bid column: {bid_col}")
            print(f"Using ask column: {ask_col}")
            
            # Rename columns to standard names
            df = df.rename(columns={
                timestamp_col: 'timestamp',
                bid_col: 'bid',
                ask_col: 'ask'
            })
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            
            # Ensure bid/ask are numeric
            df['bid'] = pd.to_numeric(df['bid'], errors='coerce')
            df['ask'] = pd.to_numeric(df['ask'], errors='coerce')
            df = df.dropna(subset=['bid', 'ask'])
            
            # Calculate derived metrics
            df['mid'] = (df['bid'] + df['ask']) / 2
            df['spread'] = df['ask'] - df['bid']
            
            print(f"Loaded {len(df)} tick records")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Price range: {df['mid'].min():.2f} to {df['mid'].max():.2f}")
            print(f"Average spread: {df['spread'].mean():.4f}")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to load tick data: {e}")
    
    def calculate_market_state(self, tick_df):
        """Calculate market state indicators for realistic L2 generation."""
        print("Calculating market state indicators...")
        
        # Rolling volatility (using log returns)
        tick_df['log_return'] = np.log(tick_df['mid'] / tick_df['mid'].shift(1))
        tick_df['volatility'] = tick_df['log_return'].rolling(
            window=self.volatility_window, min_periods=1
        ).std() * np.sqrt(252 * 24 * 60)  # Annualized
        
        # Price momentum (short-term direction)
        tick_df['momentum'] = tick_df['mid'].diff(5).fillna(0)
        
        # Spread volatility (wider spreads = more uncertainty)
        tick_df['spread_vol'] = tick_df['spread'].rolling(
            window=self.volatility_window, min_periods=1
        ).std()
        
        # Volume proxy (higher volatility = higher volume)
        tick_df['volume_proxy'] = 1000 + (tick_df['volatility'] * 50000).fillna(1000)
        
        return tick_df
    
    def generate_price_levels(self, bid, ask, spread, volatility):
        """Generate realistic price levels for order book."""
        mid = (bid + ask) / 2
        
        # Dynamic level spacing based on spread and volatility
        base_spacing = max(self.tick_size, spread * 0.3)
        vol_adjustment = 1 + (volatility * self.spread_multiplier)
        level_spacing = base_spacing * vol_adjustment
        
        # Generate bid levels (decreasing prices)
        bid_prices = []
        for i in range(self.levels):
            level_price = bid - (i * level_spacing)
            # Round to tick size
            level_price = round(level_price / self.tick_size) * self.tick_size
            bid_prices.append(level_price)
        
        # Generate ask levels (increasing prices)
        ask_prices = []
        for i in range(self.levels):
            level_price = ask + (i * level_spacing)
            # Round to tick size
            level_price = round(level_price / self.tick_size) * self.tick_size
            ask_prices.append(level_price)
        
        return bid_prices, ask_prices
    
    def generate_volumes(self, base_volume, momentum, volatility):
        """Generate realistic volume distribution across levels."""
        # Volume increases with volatility
        vol_multiplier = 1 + (volatility * 2)
        adjusted_base = base_volume * vol_multiplier
        
        # Generate order book imbalance based on momentum
        if abs(momentum) > 0.01:  # Only create imbalance for significant moves
            imbalance = np.tanh(momentum * 100) * self.imbalance_factor
        else:
            imbalance = 0
        
        bid_volumes = []
        ask_volumes = []
        
        for i in range(self.levels):
            # Exponential decay with random variation
            decay_factor = self.volume_decay ** i
            random_factor = self.rng.normal(1.0, 0.1)
            
            base_vol = adjusted_base * decay_factor * random_factor
            
            # Apply imbalance (positive = more bids, negative = more asks)
            bid_vol = base_vol * (1 + imbalance)
            ask_vol = base_vol * (1 - imbalance)
            
            # Ensure minimum volume and round to integers
            bid_volumes.append(max(100, int(bid_vol)))
            ask_volumes.append(max(100, int(ask_vol)))
        
        return bid_volumes, ask_volumes
    
    def generate_l2_data(self, tick_df):
        """Generate complete L2 data from tick data."""
        print("Generating L2 order book data...")
        
        l2_records = []
        
        for idx, row in tick_df.iterrows():
            try:
                # Extract current market state
                bid, ask = row['bid'], row['ask']
                spread = row['spread']
                volatility = row.get('volatility', 0.1)
                momentum = row.get('momentum', 0)
                volume_proxy = row.get('volume_proxy', 1000)
                
                # Generate price levels
                bid_prices, ask_prices = self.generate_price_levels(
                    bid, ask, spread, volatility
                )
                
                # Generate volumes
                bid_volumes, ask_volumes = self.generate_volumes(
                    volume_proxy, momentum, volatility
                )
                
                # Create L2 record
                l2_record = {
                    'timestamp': row['timestamp']
                }
                
                # Add bid levels (1-5)
                for i in range(self.levels):
                    l2_record[f'bid_price_{i+1}'] = bid_prices[i]
                    l2_record[f'bid_volume_{i+1}'] = bid_volumes[i]
                
                # Add ask levels (1-5)
                for i in range(self.levels):
                    l2_record[f'ask_price_{i+1}'] = ask_prices[i]
                    l2_record[f'ask_volume_{i+1}'] = ask_volumes[i]
                
                l2_records.append(l2_record)
                
                # Progress indicator
                if len(l2_records) % 1000 == 0:
                    print(f"Generated {len(l2_records)} L2 records...")
                    
            except Exception as e:
                print(f"Error generating L2 for row {idx}: {e}")
                continue
        
        # Convert to DataFrame
        l2_df = pd.DataFrame(l2_records)
        
        print(f"Generated {len(l2_df)} L2 records")
        return l2_df
    
    def validate_l2_data(self, l2_df):
        """Validate the generated L2 data for realism."""
        print("Validating L2 data quality...")
        
        # Check price ordering (bids decreasing, asks increasing)
        for i in range(1, self.levels):
            bid_col_curr = f'bid_price_{i}'
            bid_col_next = f'bid_price_{i+1}'
            
            price_order_ok = (l2_df[bid_col_curr] >= l2_df[bid_col_next]).all()
            if not price_order_ok:
                print(f"WARNING: Bid price ordering violated between levels {i} and {i+1}")
        
        for i in range(1, self.levels):
            ask_col_curr = f'ask_price_{i}'
            ask_col_next = f'ask_price_{i+1}'
            
            price_order_ok = (l2_df[ask_col_curr] <= l2_df[ask_col_next]).all()
            if not price_order_ok:
                print(f"WARNING: Ask price ordering violated between levels {i} and {i+1}")
        
        # Check spreads (ask_1 > bid_1)
        spread_positive = (l2_df['ask_price_1'] > l2_df['bid_price_1']).all()
        if not spread_positive:
            print("WARNING: Negative spreads detected")
        
        # Statistics
        avg_spread = (l2_df['ask_price_1'] - l2_df['bid_price_1']).mean()
        avg_bid_vol = l2_df['bid_volume_1'].mean()
        avg_ask_vol = l2_df['ask_volume_1'].mean()
        
        print(f"Validation results:")
        print(f"  Average spread: {avg_spread:.4f}")
        print(f"  Average bid volume L1: {avg_bid_vol:.0f}")
        print(f"  Average ask volume L1: {avg_ask_vol:.0f}")
        print(f"  Price range: {l2_df['bid_price_1'].min():.2f} - {l2_df['ask_price_1'].max():.2f}")
        
        return True
    
    def save_l2_data(self, l2_df, output_path):
        """Save L2 data to compressed CSV file."""
        print(f"Saving L2 data to {output_path}")
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert timestamp to string for CSV
        l2_df = l2_df.copy()
        l2_df['timestamp'] = l2_df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        # Save as compressed CSV
        with gzip.open(output_path, 'wt', newline='') as f:
            l2_df.to_csv(f, index=False, float_format='%.5f')
        
        print(f"L2 data saved successfully ({output_path.stat().st_size} bytes)")
        
        # Show sample
        print("\nSample L2 data (first 3 rows):")
        print(l2_df.head(3).to_string())
    
    def generate(self, tick_file_path, output_path):
        """Main method to generate L2 data from tick data."""
        print("=== XAUUSD L2 Data Generator ===")
        
        # Load tick data
        tick_df = self.load_tick_data(tick_file_path)
        
        # Calculate market state
        tick_df = self.calculate_market_state(tick_df)
        
        # Generate L2 data
        l2_df = self.generate_l2_data(tick_df)
        
        # Validate quality
        self.validate_l2_data(l2_df)
        
        # Save to file
        self.save_l2_data(l2_df, output_path)
        
        print("=== Generation Complete ===")
        return l2_df


def main():
    """Command line interface for L2 data generation."""
    parser = argparse.ArgumentParser(description='Generate realistic L2 order book data')
    parser.add_argument(
        '--tick-file', 
        required=True,
        help='Path to input tick data file (CSV.gz)'
    )
    parser.add_argument(
        '--output', 
        required=True,
        help='Path to output L2 data file (CSV.gz)'
    )
    parser.add_argument(
        '--levels', 
        type=int, 
        default=5,
        help='Number of price levels to generate (default: 5)'
    )
    parser.add_argument(
        '--tick-size', 
        type=float, 
        default=0.01,
        help='Minimum price increment (default: 0.01)'
    )
    
    args = parser.parse_args()
    
    # Generate L2 data
    generator = L2DataGenerator(tick_size=args.tick_size, levels=args.levels)
    generator.generate(args.tick_file, args.output)


if __name__ == '__main__':
    main()