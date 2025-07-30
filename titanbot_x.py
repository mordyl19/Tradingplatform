# === TitanBot-X: Fully Autonomous Adaptive Trading Bot for Interactive Brokers ===

from ib_insync import *
import pandas as pd
import numpy as np
import schedule
import time
import json
import os
import signal
import sys
import logging
from datetime import datetime, timedelta
import traceback
from typing import Optional, Dict, Any, Tuple

# Import configuration
from config import *

# Global variables
ib: Optional[IB] = None
daily_pnl = 0.0
start_balance = 0.0
active_positions: Dict[str, Dict[str, Any]] = {}
shutdown_flag = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BOT_LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === SIGNAL HANDLERS ===
def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    logger.info("Shutdown signal received. Closing positions and exiting...")
    shutdown_flag = True
    cleanup_and_exit()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# === HELPER FUNCTIONS ===
def log(msg: str) -> None:
    """Thread-safe logging with error handling"""
    try:
        logger.info(msg)
    except Exception as e:
        print(f"Logging error: {e}")

def connect_ib() -> bool:
    """Establish connection to Interactive Brokers with retry logic"""
    global ib
    
    for attempt in range(CONNECTION_RETRIES):
        try:
            if ib is not None:
                try:
                    ib.disconnect()
                except:
                    pass
            
            ib = IB()
            ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
            
            # Test connection with a simple request
            account_summary = ib.accountSummary()
            if account_summary:
                log("Successfully connected to Interactive Brokers")
                return True
            else:
                raise Exception("Connection test failed - no account summary received")
            
        except Exception as e:
            log(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < CONNECTION_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                log("Failed to connect after all attempts")
                return False
    
    return False

def get_account_balance() -> float:
    """Get account balance with proper error handling"""
    try:
        if not ib or not ib.isConnected():
            if not connect_ib():
                return 0.0
        
        account_summary = ib.accountSummary()
        if not account_summary:
            log("No account summary received")
            return 0.0
        
        # Find NetLiquidation value
        for item in account_summary:
            if item.tag == 'NetLiquidation':
                balance = float(item.value)
                log(f"Current account balance: ${balance:,.2f}")
                return balance
        
        log("NetLiquidation not found in account summary")
        return 0.0
        
    except Exception as e:
        log(f"Error getting account balance: {e}")
        return 0.0

def get_available_funds() -> float:
    """Get available funds for trading"""
    try:
        if not ib or not ib.isConnected():
            return 0.0
        
        account_summary = ib.accountSummary()
        if not account_summary:
            return 0.0
        
        for item in account_summary:
            if item.tag == 'AvailableFunds':
                funds = float(item.value)
                log(f"Available funds: ${funds:,.2f}")
                return funds
        
        return 0.0
        
    except Exception as e:
        log(f"Error getting available funds: {e}")
        return 0.0

def check_daily_loss_limit() -> bool:
    """Check if daily loss limit has been reached"""
    global daily_pnl, start_balance
    
    if start_balance <= 0:
        return False
    
    daily_return = daily_pnl / start_balance
    if daily_return <= DAILY_LOSS_LIMIT:
        log(f"Daily loss limit reached: {daily_return:.2%}")
        return True
    
    return False

def load_memory() -> Dict[str, Any]:
    """Load strategy memory with error handling"""
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r') as f:
                memory = json.load(f)
                log(f"Loaded memory for {len(memory)} symbols")
                return memory
    except Exception as e:
        log(f"Error loading memory: {e}")
    
    return {}

def save_memory(memory: Dict[str, Any]) -> None:
    """Save strategy memory with error handling"""
    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump(memory, f, indent=2)
        log(f"Saved memory for {len(memory)} symbols")
    except Exception as e:
        log(f"Error saving memory: {e}")

def adaptive_learning(symbol: str, pnl_pct: float) -> None:
    """Update strategy memory based on trade results"""
    try:
        memory = load_memory()
        if symbol not in memory:
            memory[symbol] = {
                "count": 0, 
                "total_return": 0.0, 
                "win_rate": 0.0, 
                "wins": 0,
                "avg_return": 0.0
            }
        
        memory[symbol]["count"] += 1
        memory[symbol]["total_return"] += pnl_pct
        
        if pnl_pct > 0:
            memory[symbol]["wins"] += 1
        
        memory[symbol]["win_rate"] = memory[symbol]["wins"] / memory[symbol]["count"]
        memory[symbol]["avg_return"] = memory[symbol]["total_return"] / memory[symbol]["count"]
        
        save_memory(memory)
        log(f"Updated memory for {symbol}: Win rate: {memory[symbol]['win_rate']:.1%}, Avg return: {memory[symbol]['avg_return']:.2%}")
        
    except Exception as e:
        log(f"Error in adaptive learning: {e}")

def evaluate_trade(symbol: str) -> bool:
    """Evaluate if we should trade this symbol based on historical performance"""
    try:
        memory = load_memory()
        stats = memory.get(symbol, {"count": 0, "total_return": 0.0, "win_rate": 0.0, "avg_return": 0.0})
        
        # Skip if we have enough data and poor performance
        if stats["count"] >= 5:
            avg_return = stats.get("avg_return", 0.0)
            win_rate = stats.get("win_rate", 0.0)
            
            if avg_return < -0.01 or win_rate < 0.3:  # Less than 30% win rate or -1% avg return
                log(f"Skipping {symbol} due to poor historical performance: WR: {win_rate:.1%}, AR: {avg_return:.2%}")
                return False
        
        return True
        
    except Exception as e:
        log(f"Error evaluating trade for {symbol}: {e}")
        return False

def record_trade(symbol: str, qty: int, entry: float, exit: float, pnl: float, result: str) -> None:
    """Record trade details with error handling"""
    try:
        # Create header if file doesn't exist
        if not os.path.exists(TRADE_LOG_FILE):
            with open(TRADE_LOG_FILE, 'w') as f:
                f.write("timestamp,symbol,quantity,entry_price,exit_price,pnl,pnl_pct,result\n")
        
        with open(TRADE_LOG_FILE, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pnl_pct = (exit - entry) / entry if entry > 0 else 0.0
            f.write(f"{timestamp},{symbol},{qty},{entry:.2f},{exit:.2f},{pnl:.2f},{pnl_pct:.4f},{result}\n")
            
    except Exception as e:
        log(f"Error recording trade: {e}")

# === MARKET FUNCTIONS ===
def get_market_data(symbol: str) -> Optional[pd.DataFrame]:
    """Get historical market data with error handling"""
    try:
        if not ib or not ib.isConnected():
            if not connect_ib():
                return None
        
        contract = Stock(symbol, 'SMART', 'USD')
        qualified_contracts = ib.qualifyContracts(contract)
        
        if not qualified_contracts:
            log(f"Failed to qualify contract for {symbol}")
            return None
        
        contract = qualified_contracts[0]
        
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='5 D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        if not bars:
            log(f"No historical data received for {symbol}")
            return None
        
        df = util.df(bars)
        log(f"Retrieved {len(df)} bars for {symbol}")
        return df
        
    except Exception as e:
        log(f"Error getting market data for {symbol}: {e}")
        return None

def calculate_volatility(df: pd.DataFrame) -> float:
    """Calculate volatility with error handling"""
    try:
        if df is None or len(df) < 2:
            return 0.0
        
        returns = df['close'].pct_change().dropna()
        if len(returns) < 2:
            return 0.0
            
        volatility = returns.std()
        return volatility if not np.isnan(volatility) else 0.0
        
    except Exception as e:
        log(f"Error calculating volatility: {e}")
        return 0.0

def get_current_price(symbol: str) -> Optional[float]:
    """Get current market price"""
    try:
        if not ib or not ib.isConnected():
            return None
        
        contract = Stock(symbol, 'SMART', 'USD')
        qualified_contracts = ib.qualifyContracts(contract)
        
        if not qualified_contracts:
            return None
        
        contract = qualified_contracts[0]
        ticker = ib.reqMktData(contract, '', False, False)
        ib.sleep(2)  # Wait for data
        
        # Try different price sources
        price = None
        if ticker.last and ticker.last > 0:
            price = ticker.last
        elif ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
            price = (ticker.bid + ticker.ask) / 2
        elif ticker.close and ticker.close > 0:
            price = ticker.close
        
        # Cancel market data subscription
        ib.cancelMktData(contract)
        
        if price:
            log(f"Current price for {symbol}: ${price:.2f}")
        
        return price
            
    except Exception as e:
        log(f"Error getting current price for {symbol}: {e}")
        return None

def get_signal(symbol: str) -> bool:
    """Generate trading signal with improved logic"""
    try:
        df = get_market_data(symbol)
        if df is None or len(df) < 20:
            log(f"Insufficient data for {symbol}")
            return False
        
        # Ensure we have required columns
        required_cols = ['close', 'volume', 'high', 'low']
        if not all(col in df.columns for col in required_cols):
            log(f"Missing required columns for {symbol}")
            return False
        
        # Simple momentum strategy with multiple confirmations
        current_price = df['close'].iloc[-1]
        sma_10 = df['close'].rolling(10).mean().iloc[-1]
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        
        # Volume confirmation
        avg_volume = df['volume'].rolling(10).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        # Price action confirmation (higher highs)
        recent_high = df['high'].rolling(5).max().iloc[-1]
        prev_high = df['high'].rolling(5).max().iloc[-6]
        
        # Signal conditions
        momentum_signal = current_price > sma_10 > sma_20
        volume_confirmation = current_volume > avg_volume * 1.2
        price_action_signal = recent_high > prev_high
        
        signal = momentum_signal and volume_confirmation and price_action_signal
        
        if signal:
            log(f"Strong signal for {symbol}: Price: ${current_price:.2f}, SMA10: ${sma_10:.2f}, SMA20: ${sma_20:.2f}")
        
        return signal
        
    except Exception as e:
        log(f"Error generating signal for {symbol}: {e}")
        return False

# === TRADE EXECUTION ===
def calculate_position_size(symbol: str, balance: float) -> int:
    """Calculate appropriate position size"""
    try:
        current_price = get_current_price(symbol)
        if not current_price or current_price <= 0:
            return 0
        
        # Calculate trade value
        trade_value = balance * TRADE_SIZE_FRACTION
        
        # Calculate number of shares
        shares = int(trade_value / current_price)
        
        # Ensure we don't exceed available funds
        available_funds = get_available_funds()
        max_shares = int((available_funds * (1 - ACCOUNT_CASH_BUFFER)) / current_price)
        
        position_size = min(shares, max_shares)
        
        if position_size > 0:
            cost = position_size * current_price
            log(f"Position size for {symbol}: {position_size} shares (${cost:,.2f})")
        
        return position_size
        
    except Exception as e:
        log(f"Error calculating position size for {symbol}: {e}")
        return 0

def place_order(symbol: str, qty: int) -> Optional[Trade]:
    """Place market order with proper error handling"""
    try:
        if qty <= 0:
            log(f"Invalid quantity for {symbol}: {qty}")
            return None
        
        if not ib or not ib.isConnected():
            if not connect_ib():
                return None
        
        contract = Stock(symbol, 'SMART', 'USD')
        qualified_contracts = ib.qualifyContracts(contract)
        
        if not qualified_contracts:
            log(f"Failed to qualify contract for {symbol}")
            return None
        
        contract = qualified_contracts[0]
        order = MarketOrder('BUY', qty)
        trade = ib.placeOrder(contract, order)
        
        # Wait for order to fill
        max_wait = 30  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            ib.sleep(1)
            if trade.orderStatus.status in ['Filled', 'Cancelled']:
                break
        
        if trade.orderStatus.status != 'Filled':
            log(f"Order for {symbol} not filled: {trade.orderStatus.status}")
            try:
                ib.cancelOrder(trade)
            except:
                pass
            return None
        
        fill_price = trade.orderStatus.avgFillPrice or 0.0
        log(f"Successfully bought {qty} shares of {symbol} at ${fill_price:.2f}")
        return trade
        
    except Exception as e:
        log(f"Error placing order for {symbol}: {e}")
        return None

def live_exit(symbol: str, qty: int, entry_price: float) -> float:
    """Execute exit strategy with trailing stop and time limit"""
    try:
        if not ib or not ib.isConnected():
            return entry_price
        
        contract = Stock(symbol, 'SMART', 'USD')
        qualified_contracts = ib.qualifyContracts(contract)
        
        if not qualified_contracts:
            return entry_price
        
        contract = qualified_contracts[0]
        max_price = entry_price
        start_time = time.time()
        hold_seconds = DEFAULT_HOLD_MINUTES * 60
        
        # Get volatility for dynamic exit
        df = get_market_data(symbol)
        vol = calculate_volatility(df) if df is not None else 0.0
        
        log(f"Starting exit monitoring for {symbol}, entry: ${entry_price:.2f}, volatility: {vol:.4f}")
        
        while time.time() - start_time < hold_seconds:
            if shutdown_flag:
                break
                
            current_price = get_current_price(symbol)
            if not current_price:
                ib.sleep(5)
                continue
            
            max_price = max(max_price, current_price)
            drawdown = (max_price - current_price) / max_price if max_price > 0 else 0
            profit_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            
            # Exit conditions
            if drawdown >= TRAIL_STOP_PCT:
                log(f"Trailing stop triggered for {symbol}: {drawdown:.2%} drawdown")
                break
                
            if vol > 0 and profit_pct >= VOL_MULTIPLIER * vol:
                log(f"Profit target reached for {symbol}: {profit_pct:.2%}")
                break
            
            ib.sleep(5)
        
        # Place sell order
        sell_order = MarketOrder('SELL', qty)
        sell_trade = ib.placeOrder(contract, sell_order)
        
        # Wait for sell order to fill
        max_wait = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            ib.sleep(1)
            if sell_trade.orderStatus.status in ['Filled', 'Cancelled']:
                break
        
        if sell_trade.orderStatus.status == 'Filled':
            exit_price = sell_trade.orderStatus.avgFillPrice or current_price
            log(f"Successfully sold {qty} shares of {symbol} at ${exit_price:.2f}")
            return exit_price
        else:
            log(f"Sell order for {symbol} not filled: {sell_trade.orderStatus.status}")
            return current_price or entry_price
            
    except Exception as e:
        log(f"Error in live exit for {symbol}: {e}")
        return entry_price

# === CORE STRATEGY ===
def run_trading_day() -> None:
    """Execute daily trading strategy with comprehensive error handling"""
    global daily_pnl, start_balance
    
    try:
        log("Starting TitanBot trading session...")
        
        # Initialize daily tracking
        if start_balance == 0:
            start_balance = get_account_balance()
            daily_pnl = 0.0
            if start_balance == 0:
                log("Failed to get account balance. Aborting trading session.")
                return
        
        # Check daily loss limit
        if check_daily_loss_limit():
            log("Daily loss limit reached. Skipping trading.")
            return
        
        balance = get_account_balance()
        if balance <= MIN_BALANCE:
            log(f"Insufficient balance for trading: ${balance:,.2f} (min: ${MIN_BALANCE:,.2f})")
            return
        
        available_funds = get_available_funds()
        if available_funds <= MIN_AVAILABLE_FUNDS:
            log(f"Insufficient available funds: ${available_funds:,.2f} (min: ${MIN_AVAILABLE_FUNDS:,.2f})")
            return
        
        # Trading symbols from configuration
        tickers = TRADING_SYMBOLS
        trades_executed = 0
        
        log(f"Starting trading with balance: ${balance:,.2f}, available: ${available_funds:,.2f}")
        
        for symbol in tickers:
            if shutdown_flag or trades_executed >= MAX_TRADES_PER_DAY:
                break
                
            try:
                log(f"Analyzing {symbol}...")
                
                # Check if we should trade this symbol
                if not get_signal(symbol):
                    log(f"No signal for {symbol}")
                    continue
                    
                if not evaluate_trade(symbol):
                    continue
                
                # Calculate position size
                qty = calculate_position_size(symbol, balance)
                if qty <= 0:
                    log(f"Invalid position size for {symbol}: {qty}")
                    continue
                
                # Place entry order
                trade = place_order(symbol, qty)
                if not trade:
                    continue
                
                entry_price = trade.orderStatus.avgFillPrice
                if not entry_price or entry_price <= 0:
                    log(f"Invalid entry price for {symbol}: {entry_price}")
                    continue
                
                # Execute exit strategy
                exit_price = live_exit(symbol, qty, entry_price)
                
                # Calculate results
                pnl = (exit_price - entry_price) * qty
                pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
                result = "win" if pnl > 0 else "loss"
                
                # Update tracking
                daily_pnl += pnl
                trades_executed += 1
                
                # Record trade
                record_trade(symbol, qty, entry_price, exit_price, pnl, result)
                adaptive_learning(symbol, pnl_pct)
                
                log(f"{symbol} trade completed: {result}, PnL: ${pnl:.2f} ({pnl_pct:.2%})")
                
                # Brief pause between trades
                time.sleep(10)
                
            except Exception as e:
                log(f"Error trading {symbol}: {e}")
                continue
        
        # Session summary
        current_balance = get_account_balance()
        session_pnl = current_balance - balance
        daily_return = daily_pnl / start_balance if start_balance > 0 else 0
        
        log(f"Trading session completed.")
        log(f"Trades executed: {trades_executed}")
        log(f"Session PnL: ${session_pnl:.2f}")
        log(f"Daily PnL: ${daily_pnl:.2f} ({daily_return:.2%})")
        log(f"Current balance: ${current_balance:,.2f}")
        
    except Exception as e:
        log(f"Error in trading session: {e}")
        log(f"Traceback: {traceback.format_exc()}")

def cleanup_and_exit() -> None:
    """Clean up resources and exit gracefully"""
    global ib
    
    try:
        log("Cleaning up and exiting...")
        
        # Cancel any pending orders
        if ib and ib.isConnected():
            try:
                open_orders = ib.openOrders()
                for order in open_orders:
                    ib.cancelOrder(order)
                    log(f"Cancelled order: {order}")
            except Exception as e:
                log(f"Error cancelling orders: {e}")
            
            # Disconnect from IB
            try:
                ib.disconnect()
                log("Disconnected from Interactive Brokers")
            except Exception as e:
                log(f"Error disconnecting: {e}")
        
    except Exception as e:
        log(f"Error during cleanup: {e}")
    finally:
        sys.exit(0)

# === SCHEDULER ===
def init_scheduler() -> None:
    """Initialize trading schedule"""
    # Schedule based on configuration
    for day, time_str in TRADING_SCHEDULE.items():
        getattr(schedule.every(), day).at(time_str).do(run_trading_day)
    
    log(f"Trading schedule initialized: {TRADING_SCHEDULE}")

def health_check() -> bool:
    """Perform system health check"""
    try:
        # Check IB connection
        if not ib or not ib.isConnected():
            log("Health check failed: No IB connection")
            return False
        
        # Check account balance
        balance = get_account_balance()
        if balance <= 0:
            log("Health check failed: Unable to get account balance")
            return False
        
        log("Health check passed")
        return True
        
    except Exception as e:
        log(f"Health check error: {e}")
        return False

def main() -> None:
    """Main execution loop"""
    global shutdown_flag
    
    try:
        log("TitanBot-X initializing...")
        log(f"Configuration: Loss limit: {DAILY_LOSS_LIMIT:.1%}, Trade size: {TRADE_SIZE_FRACTION:.1%}")
        log(f"Hold time: {DEFAULT_HOLD_MINUTES}min, Trail stop: {TRAIL_STOP_PCT:.1%}")
        
        # Initial connection
        if not connect_ib():
            log("Failed to establish initial connection. Exiting.")
            return
        
        # Initialize scheduler
        init_scheduler()
        
        log("TitanBot-X initialized successfully. Waiting for scheduled trades...")
        
        # Main loop
        last_health_check = time.time()
        
        while not shutdown_flag:
            try:
                schedule.run_pending()
                time.sleep(MAIN_LOOP_SLEEP)
                
                # Periodic health check
                if time.time() - last_health_check > HEALTH_CHECK_INTERVAL:
                    if not health_check():
                        log("Health check failed. Attempting to reconnect...")
                        connect_ib()
                    last_health_check = time.time()
                    
            except KeyboardInterrupt:
                log("Keyboard interrupt received")
                break
            except Exception as e:
                log(f"Error in main loop: {e}")
                time.sleep(30)  # Wait before continuing
                
    except Exception as e:
        log(f"Fatal error in main: {e}")
        log(f"Traceback: {traceback.format_exc()}")
    finally:
        cleanup_and_exit()

if __name__ == '__main__':
    main()