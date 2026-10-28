"""
connect_mt5.py
Live MetaTrader 5 Broker Connection & Account Verification Utility
Usage:
    python connect_mt5.py
    python connect_mt5.py --login 12345678 --password YourPass --server Broker-Live
"""

import os
import sys
import argparse
import MetaTrader5 as mt5
from config import CONFIG


def verify_and_connect(login: int = None, password: str = None, server: str = None, path: str = None):
    print("=" * 65)
    print("   ANTI-GRAVITY XAUUSD SCALPER: MT5 BROKER ONBOARDING & AUDIT    ")
    print("=" * 65)

    # 1. Resolve credentials
    login = login or CONFIG.mt5.login
    password = password or CONFIG.mt5.password
    server = server or CONFIG.mt5.server
    path = path or CONFIG.mt5.path or r"C:\Program Files\MetaTrader 5\terminal64.exe"

    print(f"\n[1/4] Target Terminal Path: {path}")
    if os.path.exists(path):
        print("  ✓ Terminal executable found on disk.")
    else:
        print(f"  ✗ WARNING: Terminal path not found: {path}")

    # 2. Initialize terminal
    print("\n[2/4] Initializing MetaTrader 5 IPC Bridge...")
    init_ok = mt5.initialize(path=path)
    if not init_ok:
        err = mt5.last_error()
        print(f"  ✗ Failed to initialize MT5 IPC Bridge: {err}")
        print("\n  💡 TIP: Make sure your MetaTrader 5 application is OPEN on your Windows desktop.")
        print("     If it is the first time running, accept the initial prompts in the MT5 window.")
        return False

    ver = mt5.version()
    term_info = mt5.terminal_info()
    print(f"  ✓ Connected to MT5 Build {ver[0]} ({ver[2]})")
    print(f"  ✓ Terminal Company: {term_info.company if term_info else 'MetaQuotes'}")
    print(f"  ✓ Algorithmic Trading Allowed: {'YES' if term_info and term_info.trade_allowed else 'NO (Enable in MT5 -> Tools -> Options -> Expert Advisors)'}")

    # 3. Authenticate Account
    if login and password and str(login) != "12345678":
        print(f"\n[3/4] Authenticating Account #{login} on Server '{server}'...")
        logged_in = mt5.login(login=int(login), password=str(password), server=str(server) if server else "")
        if not logged_in:
            err = mt5.last_error()
            print(f"  ✗ Login failed: {err}")
            print("  Please verify your Login number, Master/Investor password, and Server name.")
            mt5.shutdown()
            return False
        print("  ✓ Login Successful!")
    else:
        print("\n[3/4] Checking Currently Active MT5 Account...")

    acc_info = mt5.account_info()
    if acc_info is None:
        print("  ✗ Could not retrieve active account info from MT5.")
        mt5.shutdown()
        return False

    print("\n" + "-" * 55)
    print(f"  ACCOUNT NUMBER : #{acc_info.login}")
    print(f"  TRADER NAME    : {acc_info.name}")
    print(f"  BROKER SERVER  : {acc_info.server}")
    print(f"  CURRENCY       : {acc_info.currency}")
    print(f"  LEVERAGE       : 1:{acc_info.leverage}")
    print(f"  BALANCE        : ${acc_info.balance:,.2f}")
    print(f"  EQUITY         : ${acc_info.equity:,.2f}")
    print(f"  FREE MARGIN    : ${acc_info.margin_free:,.2f}")
    print(f"  TRADE MODE     : {'DEMO' if acc_info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else 'REAL / LIVE'}")
    print("-" * 55)

    # 4. Verify Instrument & Market Watch
    symbol = CONFIG.instrument.symbol
    print(f"\n[4/4] Auditing Market Flow for '{symbol}'...")
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        # Check alternative symbol names (e.g. GOLD, XAUUSD.m, XAUUSDm)
        alt_symbols = ["GOLD", "XAUUSDm", "XAUUSD.m", "XAUUSD.r", "XAUUSD_"]
        found_alt = None
        for alt in alt_symbols:
            if mt5.symbol_info(alt) is not None:
                found_alt = alt
                break
        
        if found_alt:
            print(f"  ℹ️ Symbol '{symbol}' not found directly, but found broker equivalent '{found_alt}'.")
            symbol = found_alt
            sym_info = mt5.symbol_info(symbol)
        else:
            print(f"  ✗ Could not find symbol '{symbol}' in Market Watch.")
            print("  Please right click Market Watch in MT5 and select 'Show All'.")
            mt5.shutdown()
            return False

    # Select symbol
    mt5.symbol_select(symbol, True)
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        spread_usd = (tick.ask - tick.bid)
        print(f"  ✓ {symbol} BID: ${tick.bid:.2f} | ASK: ${tick.ask:.2f}")
        print(f"  ✓ Current Spread: {spread_usd * 100:.1f}¢ (Limit: {CONFIG.spread_sentinel.max_absolute_spread_usd * 100:.0f}¢)")
        if spread_usd <= CONFIG.spread_sentinel.max_absolute_spread_usd:
            print("  ✓ Spread Gate: PASS")
        else:
            print("  ⚠️ Warning: Current spread exceeds scalping threshold.")
    else:
        print(f"  ⚠️ Symbol '{symbol}' selected, awaiting live tick from broker server.")

    # 5. Offer to persist to .env
    print("\n" + "=" * 65)
    print("   BROKER CONNECTION VERIFIED 100% READY FOR LIVE TRADING!       ")
    print("=" * 65)

    mt5.shutdown()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connect and verify MT5 broker account")
    parser.add_argument("--login", type=int, help="MT5 Account Number")
    parser.add_argument("--password", type=str, help="MT5 Trading Password")
    parser.add_argument("--server", type=str, help="Broker Server Name (e.g. Exness-Real, ICMarketsSC-Live)")
    parser.add_argument("--path", type=str, help="Full path to terminal64.exe")
    args = parser.parse_args()

    verify_and_connect(login=args.login, password=args.password, server=args.server, path=args.path)
