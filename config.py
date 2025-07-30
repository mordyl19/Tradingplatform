# === TitanBot-X Configuration ===

# Trading Configuration
ACCOUNT_CASH_BUFFER = 0.05          # Preserve 5% cash buffer
DAILY_LOSS_LIMIT = -0.05            # Maximum daily loss (-5%)
TRADE_SIZE_FRACTION = 0.2           # Percentage of balance per trade (20%)
DEFAULT_HOLD_MINUTES = 90           # Default hold time in minutes
TRAIL_STOP_PCT = 0.03               # Trailing stop percentage (3%)
VOL_MULTIPLIER = 1.5                # Volatility target multiplier
MIN_BALANCE = 1000.0                # Minimum account balance for trading
MIN_AVAILABLE_FUNDS = 100.0         # Minimum available funds
MAX_TRADES_PER_DAY = 3              # Maximum trades per session

# Connection Configuration
IB_HOST = '127.0.0.1'               # Interactive Brokers TWS/Gateway host
IB_PORT = 7497                      # Interactive Brokers TWS/Gateway port (7497 for TWS, 4002 for Gateway)
CLIENT_ID = 1                       # Unique client ID
CONNECTION_RETRIES = 3              # Maximum connection retry attempts
RETRY_DELAY = 5                     # Delay between retries in seconds

# File Configuration
MEMORY_FILE = "strategy_history.json"
TRADE_LOG_FILE = "trade_log.csv"
BOT_LOG_FILE = "titanbot.log"

# Trading Symbols - Diversified Portfolio
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

# Schedule Configuration
TRADING_SCHEDULE = {
    "monday": "09:45",
    "tuesday": "09:45", 
    "wednesday": "09:45",
    "thursday": "09:45",
    "friday": "09:45"
}

# Health Check Configuration
HEALTH_CHECK_INTERVAL = 300         # Health check every 5 minutes
MAIN_LOOP_SLEEP = 10                # Main loop sleep time in seconds

# Signal Generation Parameters
MIN_DATA_POINTS = 20                # Minimum data points for signal generation
SMA_SHORT_PERIOD = 10               # Short-term moving average period
SMA_LONG_PERIOD = 20                # Long-term moving average period
VOLUME_MULTIPLIER = 1.2             # Volume confirmation multiplier
LOOKBACK_PERIOD = 5                 # Price action lookback period

# Performance Evaluation Thresholds
MIN_TRADES_FOR_EVALUATION = 5       # Minimum trades before evaluating performance
MIN_WIN_RATE = 0.3                  # Minimum win rate (30%)
MIN_AVG_RETURN = -0.01              # Minimum average return (-1%)

# Order Execution Timeouts
ORDER_FILL_TIMEOUT = 30             # Maximum wait time for order fill
MARKET_DATA_WAIT = 2                # Wait time for market data
EXIT_MONITORING_SLEEP = 5           # Sleep time during exit monitoring