# XAUUSD Scalping System

A production-grade, institution-level low-latency intraday scalping system for XAUUSD (Gold/USD) trading. This system combines advanced feature engineering, machine learning models, and sophisticated risk management to execute high-frequency trading strategies on gold futures.

## 🎯 Overview

This system is designed for professional traders and institutions seeking to implement systematic scalping strategies on the XAUUSD pair. It features:

- **Ultra-low latency**: Optimized for sub-millisecond execution times
- **Advanced ML Models**: Deep learning architectures for price prediction
- **Comprehensive Risk Management**: Multi-layered risk controls and position sizing
- **Professional Backtesting**: Institutional-grade backtesting with realistic market simulation
- **Real-time Processing**: Tick-by-tick data processing with Level 2 order book analysis
- **Production Ready**: Docker containerization and cloud deployment support

## 🏗️ Architecture
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Feature Layer  │    │   Model Layer   │
│                 │    │                 │    │                 │
│ • Tick Data     │───▶│ • Technical     │───▶│ • Neural Nets   │
│ • L2 OrderBook  │    │ • Microstructure│    │ • Ensemble      │
│ • News Events   │    │ • Sentiment     │    │ • Predictions   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
│                       │                       │
▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Risk Manager   │    │ Exchange Sim    │    │   Reporting     │
│                 │    │                 │    │                 │
│ • Position Size │    │ • Order Match   │    │ • Performance   │
│ • Stop Loss     │    │ • Slippage      │    │ • Analytics     │
│ • Exposure      │    │ • Commission    │    │ • Visualization │
└─────────────────┘    └─────────────────┘    └─────────────────┘

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- NVIDIA GPU (recommended for training)
- 32GB+ RAM (for large datasets)
- SSD storage (for I/O performance)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/xauusd-scalper.git
   cd xauusd-scalper

Set up Python environment
bash# Using conda (recommended)
conda env create -f environment.yml
conda activate xauusd-scalper

# Or using pip
pip install -r requirements.txt

Verify installation
bashpython src/main.py --help


Docker Setup (Alternative)
bash# Build the Docker image
docker-compose build

# Run the system
docker-compose up -d

# Execute commands inside container
docker-compose exec app python src/main.py --help
📊 Usage
Training a Model
Train a new model using historical XAUUSD data:
bash# Basic training
python src/main.py train --data-path data/historical/ticks/

# Advanced training with custom parameters
python src/main.py train \
  --data-path data/historical/ticks/ \
  --model-name xauusd_v2 \
  --epochs 200 \
  --batch-size 1024 \
  --learning-rate 0.001 \
  --validation-split 0.2
Training Options:

--data-path, -d: Path to training data directory
--model-name, -m: Name for the trained model (default: xauusd_model)
--epochs, -e: Number of training epochs
--batch-size, -b: Training batch size
--learning-rate, -lr: Learning rate for optimizer
--validation-split, -vs: Validation data split ratio
--save-path, -s: Directory to save the trained model

Running Backtests
Execute comprehensive backtests with trained models:
bash# Basic backtest
python src/main.py backtest --model-path models/xauusd_model.keras

# Advanced backtest with custom parameters
python src/main.py backtest \
  --model-path models/xauusd_v2.keras \
  --data-path data/historical/ticks/ \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --initial-balance 100000 \
  --output-dir reports/backtest_2023 \
  --show-plots
Backtest Options:

--model-path, -m: Path to trained model file (required)
--data-path, -d: Path to backtest data directory
--start-date, -sd: Start date for backtest (YYYY-MM-DD)
--end-date, -ed: End date for backtest (YYYY-MM-DD)
--initial-balance, -ib: Initial capital for backtest
--output-dir, -o: Directory to save reports
--no-plots: Skip plot generation for faster execution
--show-plots: Display plots interactively

Configuration Management
Validate and inspect your configuration:
bash# Validate configuration file
python src/main.py validate-config

# Use custom config file
python src/main.py --config configs/production.yml backtest --model-path models/prod_model.keras
⚙️ Configuration
The system is configured via YAML files in the configs/ directory. Here's the structure:
yaml# configs/config.yml
data:
  historical_data_path: "data/historical/"
  live_data_path: "data/live/"
  tick_file_pattern: "XAUUSD_*_ticks.csv.gz"
  orderbook_file_pattern: "XAUUSD_*_l2.csv.gz"

model:
  architecture: "lstm_attention"
  sequence_length: 100
  features_to_use: ["minimal", "extended"]
  target_horizon: 5  # seconds
  
backtesting:
  initial_balance: 100000.0
  commission_per_lot: 5.0
  spread_multiplier: 1.2
  slippage_model: "linear"
  
risk_management:
  max_position_size: 0.02  # 2% of balance
  max_daily_loss_pct: 0.05  # 5% daily stop
  kelly_fraction: 0.25
  atr_periods: 14
  
trading:
  min_prediction_confidence: 0.55
  position_hold_time_max: 300  # seconds
  news_blackout_minutes: 30
📈 Performance Metrics
The system tracks comprehensive performance metrics:
Core Metrics

Total Return: Overall portfolio return percentage
Sharpe Ratio: Risk-adjusted return measure
Sortino Ratio: Downside deviation-adjusted return
Calmar Ratio: Return to maximum drawdown ratio
Maximum Drawdown: Largest peak-to-trough decline
Win Rate: Percentage of profitable trades
Profit Factor: Ratio of gross profit to gross loss

Advanced Analytics

Value at Risk (VaR): Potential loss estimation
Expected Shortfall: Tail risk measure
Information Ratio: Excess return per unit of active risk
Trade Distribution: Statistical analysis of trade outcomes
Exposure Analysis: Time-based position exposure

Visualization Reports
Generated reports include:

Equity Curve: Portfolio value over time
Drawdown Analysis: Underwater equity curves
Trade Distribution: Histograms and box plots of PnL
Risk Metrics Dashboard: Real-time risk monitoring
Performance Attribution: Source of returns analysis

🔧 Development
Project Structure
xauusd_scalper/
├── src/
│   ├── data_ingestion/     # Data loading and preprocessing
│   ├── features/           # Feature engineering pipeline
│   ├── models/             # ML model architectures and training
│   ├── backtesting/        # Backtesting engine and simulation
│   ├── risk/               # Risk management and position sizing
│   └── utils/              # Logging, helpers, and utilities
├── configs/                # Configuration files
├── data/                   # Historical and live data
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation and whitepapers
└── reports/                # Generated backtest reports
Key Components
Data Ingestion (src/data_ingestion/)

DataLoader: Handles tick data and L2 order book loading
Data Alignment: Synchronizes multiple data sources
Compression: Efficient storage and retrieval of large datasets

Feature Engineering (src/features/)

FeaturePipeline: Orchestrates feature computation
Technical Indicators: 50+ technical analysis features
Microstructure Features: Order book imbalance, flow toxicity
Alternative Data: News sentiment, economic events

Model Architecture (src/models/)

LSTM Networks: Sequential pattern recognition
Attention Mechanisms: Focus on relevant time periods
Ensemble Methods: Multiple model combination
Online Learning: Adaptive model updates

Backtesting Engine (src/backtesting/)

Tick-by-Tick Simulation: Highest fidelity backtesting
Realistic Execution: Latency, slippage, and commission modeling
Market Impact: Price impact from large orders
Multi-Asset Support: Portfolio-level backtesting

Risk Management (src/risk/)

Position Sizing: Kelly criterion and volatility-based sizing
Stop Loss: Dynamic and trailing stop mechanisms
Exposure Limits: Sector and asset concentration limits
Real-time Monitoring: Live risk metric computation

Testing
Run the comprehensive test suite:
bash# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/ --cov-report=html

# Run specific test categories
pytest tests/test_features.py -v
pytest tests/test_backtesting_engine.py -v
pytest tests/test_risk_manager.py -v
Code Quality
The project maintains high code quality standards:
bash# Code formatting
black src/ tests/

# Import sorting
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
🚀 Deployment
Production Deployment

Build production Docker image
bashdocker build -f Dockerfile.prod -t xauusd-scalper:prod .

Deploy to cloud
bash# AWS ECS
aws ecs deploy --cluster prod-cluster --service xauusd-scalper

# Kubernetes
kubectl apply -f k8s/deployment.yml

Set up monitoring
bash# Prometheus metrics
docker-compose -f docker-compose.monitoring.yml up -d


Performance Optimization
For maximum performance:

Use SSD storage for data files
Enable GPU acceleration for model training
Optimize memory usage with data streaming
Configure CPU affinity for critical processes
Implement connection pooling for data sources

📚 Documentation
Additional Resources

Whitepaper: Detailed methodology and research
API Documentation: Complete API reference
Deployment Guide: Production deployment instructions
Performance Tuning: Optimization techniques

Research Papers
Key academic references:

"High-Frequency Trading and Price Discovery" - Brogaard et al.
"The Flash Crash: High-Frequency Trading in an Electronic Market" - Kirilenko et al.
"Machine Learning for Market Microstructure" - Sirignano & Cont

⚠️ Risk Disclosure
Important: This system is designed for sophisticated traders and institutions. Trading financial instruments involves substantial risk of loss. Past performance is not indicative of future results. Please ensure you understand the risks before using this system with real capital.
Risk Warnings

High Leverage: The system may use leverage which amplifies both gains and losses
Market Risk: All trading strategies are subject to market risk
Technology Risk: System failures can result in significant losses
Regulatory Risk: Trading regulations may change affecting system operation

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🤝 Contributing
We welcome contributions from the trading and quantitative finance community:

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

Contribution Guidelines

Follow PEP 8 style guidelines
Add comprehensive tests for new features
Update documentation for API changes
Ensure all tests pass before submitting

📞 Support
For support and questions:

Issues: GitHub Issues
Discussions: GitHub Discussions
Email: support@your-org.com

🙏 Acknowledgments

QuantLib: Financial mathematics library
NumPy/Pandas: Data manipulation and analysis
TensorFlow/Keras: Machine learning framework
Matplotlib/Seaborn: Visualization libraries
Click: Command-line interface framework


Disclaimer: This software is provided for educational and research purposes. The authors are not responsible for any financial losses incurred through the use of this system. Please trade responsibly.