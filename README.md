# TitanBot-X: Autonomous Adaptive Trading Bot

## Overview

TitanBot-X is a fully autonomous, adaptive trading bot designed for Interactive Brokers. It employs machine learning-like adaptive strategies, sophisticated risk management, and robust error handling to execute algorithmic trading strategies.

## Features

- **Autonomous Trading**: Executes trades automatically based on technical signals
- **Adaptive Learning**: Learns from past performance to improve future trading decisions
- **Risk Management**: Implements daily loss limits, trailing stops, and position sizing
- **Robust Error Handling**: Comprehensive error handling and connection management
- **Flexible Configuration**: Easy-to-modify configuration system
- **Comprehensive Logging**: Detailed logging and trade recording
- **Health Monitoring**: Automated system health checks and recovery

## Prerequisites

1. **Interactive Brokers Account**: Active trading account with sufficient funds
2. **TWS or IB Gateway**: Interactive Brokers Trader Workstation or Gateway installed and running
3. **Python 3.8+**: Python environment with required packages
4. **API Permissions**: Enable API trading in your IB account

## Installation

1. **Clone or download the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Interactive Brokers**:
   - Enable API connections in TWS/Gateway
   - Set API port (default: 7497 for TWS, 4002 for Gateway)
   - Ensure "Read-Only API" is disabled if you want to trade

## Configuration

Edit `config.py` to customize the bot's behavior:

### Trading Parameters
- `TRADE_SIZE_FRACTION`: Percentage of balance per trade (default: 20%)
- `DAILY_LOSS_LIMIT`: Maximum daily loss limit (default: -5%)
- `TRAIL_STOP_PCT`: Trailing stop percentage (default: 3%)
- `DEFAULT_HOLD_MINUTES`: Maximum hold time per trade (default: 90 minutes)

### Risk Management
- `MIN_BALANCE`: Minimum account balance required for trading
- `ACCOUNT_CASH_BUFFER`: Cash buffer to maintain (default: 5%)
- `MAX_TRADES_PER_DAY`: Maximum trades per session (default: 3)

### Connection Settings
- `IB_HOST`: Interactive Brokers host (default: localhost)
- `IB_PORT`: IB API port (7497 for TWS, 4002 for Gateway)
- `CLIENT_ID`: Unique client identifier

### Trading Symbols
Modify `TRADING_SYMBOLS` list to change which stocks/ETFs to trade:
```python
TRADING_SYMBOLS = [
    "SPY",      # S&P 500 ETF
    "QQQ",      # NASDAQ 100 ETF
    "TQQQ",     # 3x Leveraged NASDAQ ETF
    "TSLA",     # Tesla
    "NVDA",     # NVIDIA
    "AAPL",     # Apple
    "MSFT",     # Microsoft
    "GOOGL",    # Google
]
```

## Usage

### Basic Usage
```bash
python titanbot_x.py
```

### Background Execution
```bash
nohup python titanbot_x.py > output.log 2>&1 &
```

### With Virtual Environment
```bash
source venv/bin/activate
python titanbot_x.py
```

## Trading Strategy

The bot implements a momentum-based strategy with multiple confirmations:

1. **Price Momentum**: Current price above short-term moving average above long-term moving average
2. **Volume Confirmation**: Current volume significantly above average volume
3. **Price Action**: Recent highs breaking previous highs
4. **Adaptive Learning**: Historical performance evaluation for each symbol

### Entry Conditions
- All technical indicators align for a bullish signal
- Symbol passes historical performance evaluation
- Sufficient account balance and available funds
- Within daily trade limits

### Exit Conditions
- Trailing stop loss triggered (3% default)
- Profit target reached (based on volatility)
- Maximum hold time exceeded (90 minutes default)
- Market close or emergency shutdown

## Risk Management

### Daily Loss Limit
- Automatically stops trading if daily loss exceeds configured limit
- Tracks profit/loss across all trades in a session

### Position Sizing
- Calculates position size based on account balance and risk parameters
- Respects available funds and cash buffer requirements
- Never risks more than configured percentage per trade

### Trailing Stops
- Dynamic trailing stop loss follows price movements
- Locks in profits while allowing for continued upside

## File Structure

```
titanbot-x/
├── titanbot_x.py          # Main trading bot code
├── config.py              # Configuration parameters
├── requirements.txt       # Python dependencies
├── README.md             # This documentation
├── strategy_history.json # Adaptive learning memory (auto-created)
├── trade_log.csv         # Trade history log (auto-created)
└── titanbot.log          # System log file (auto-created)
```

## Logging and Monitoring

### Log Files
- `titanbot.log`: System events, errors, and trading activities
- `trade_log.csv`: Detailed trade records with P&L tracking
- `strategy_history.json`: Adaptive learning data for each symbol

### Log Levels
- System startup and shutdown events
- Connection status and health checks
- Trade execution details
- Error messages and recovery actions
- Performance statistics

## Safety Features

### Connection Management
- Automatic reconnection on disconnection
- Connection health monitoring
- Graceful shutdown handling

### Error Recovery
- Robust exception handling throughout
- Automatic retry mechanisms
- Safe fallback behaviors

### Emergency Shutdown
- Signal handling for clean shutdown (Ctrl+C, SIGTERM)
- Automatic order cancellation on exit
- Resource cleanup and disconnection

## Performance Tracking

### Adaptive Learning
- Tracks win rate and average returns per symbol
- Automatically excludes poorly performing symbols
- Continuously updates performance metrics

### Trade Analytics
- Detailed trade logs with entry/exit prices
- P&L tracking and percentage returns
- Session and daily performance summaries

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Ensure TWS/Gateway is running
   - Check API settings in TWS
   - Verify port number and permissions

2. **No Trades Executed**
   - Check if signals are being generated
   - Verify minimum balance requirements
   - Review symbol performance history

3. **Permission Errors**
   - Ensure API trading is enabled
   - Check "Read-Only API" setting
   - Verify account permissions

### Debug Mode
Add debug logging by modifying the logging configuration:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Disclaimer

⚠️ **IMPORTANT**: This trading bot is for educational and research purposes. Trading involves significant financial risk. Past performance does not guarantee future results. Always:

- Test thoroughly in paper trading mode first
- Start with small position sizes
- Monitor the bot's performance closely
- Understand the risks involved in algorithmic trading
- Comply with all applicable regulations

The authors are not responsible for any financial losses incurred through the use of this software.

## License

This project is open source. Use at your own risk.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for error messages
3. Ensure all prerequisites are met
4. Test with paper trading first