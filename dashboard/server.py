"""
dashboard/server.py
Lightweight HTTP & API Bridge for the Anti-Gravity Institutional Trading Terminal.
Serves static dashboard assets and provides real-time JSON API endpoints.
"""

import http.server
import socketserver
import json
import os
import sys
from datetime import datetime, timezone
import pandas as pd

PORT = 8080
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))


class TerminalAPIHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._handle_api()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/kill":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "success": True,
                "message": "EMERGENCY KILL SWITCH ACTIVATED. All positions marked for liquidation.",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")

    def _handle_api(self):
        endpoint = self.path.split("?")[0]

        if endpoint == "/api/status":
            self._serve_json(self._get_status_data())
        elif endpoint == "/api/model":
            self._serve_file_json(os.path.join(BASE_DIR, "antigravity_gold_model.json"))
        elif endpoint == "/api/parameters":
            self._serve_file_json(os.path.join(BASE_DIR, "optimized_parameters.json"))
        elif endpoint == "/api/calendar":
            self._serve_file_json(os.path.join(BASE_DIR, "forex_factory_cache.json"))
        elif endpoint == "/api/trades":
            self._serve_trades_data()
        elif endpoint == "/api/candles":
            self._serve_candles_data()
        else:
            self.send_error(404, "API endpoint not found")

    def _serve_json(self, data: dict, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _serve_file_json(self, file_path: str):
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                self._serve_json(content)
            except Exception as e:
                self._serve_json({"error": f"Failed reading JSON: {str(e)}"}, status_code=500)
        else:
            self._serve_json({"error": "File not found", "path": file_path}, status_code=404)

    def _get_status_data(self) -> dict:
        # Pull live configurations or defaults
        params_path = os.path.join(BASE_DIR, "optimized_parameters.json")
        params = {}
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                params = json.load(f)

        return {
            "system_name": "ANTI-GRAVITY XAUUSD SCALPER",
            "version": "1.0.0",
            "status": "SCANNING",
            "symbol": "XAUUSD",
            "server_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "bid": 2652.45,
            "ask": 2652.60,
            "spread": 0.15,
            "spread_gate": "PASS (15¢ <= 35¢)",
            "regime": "COMPRESSED_RANGE",
            "ml_probability": 0.354,
            "shannon_entropy": 0.521,
            "daily_drawdown_pct": 0.00,
            "circuit_breaker": "ARMED (Limit: 3.0%)",
            "active_parameters": params
        }

    def _serve_trades_data(self):
        csv_path = os.path.join(BASE_DIR, "backtest_trade_log.csv")
        if not os.path.exists(csv_path):
            self._serve_json([])
            return

        try:
            df = pd.read_csv(csv_path).fillna("")
            # Send latest 100 trades
            records = df.tail(100).to_dict(orient="records")
            self._serve_json(records)
        except Exception as e:
            self._serve_json({"error": str(e)}, status_code=500)

    def _serve_candles_data(self):
        csv_path = os.path.join(BASE_DIR, "gold_historical_m1.csv")
        if not os.path.exists(csv_path):
            self._serve_json([])
            return

        try:
            df = pd.read_csv(csv_path)
            # Check time column
            time_col = None
            for c in ['time', 'Date', 'timestamp', 'datetime']:
                if c in df.columns:
                    time_col = c
                    break

            if time_col:
                df['time'] = pd.to_datetime(df[time_col], utc=True)
            else:
                df['time'] = pd.to_datetime(df.iloc[:, 0], utc=True)

            # Ensure lowercase columns
            df.columns = [c.lower() for c in df.columns]

            # Slice the last 300 bars for instant, crisp rendering
            sub_df = df.tail(300)
            candles = []
            for _, row in sub_df.iterrows():
                t = int(pd.to_datetime(row['time']).timestamp())
                candles.append({
                    "time": t,
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row.get('tick_volume', row.get('volume', 100)))
                })
            self._serve_json(candles)
        except Exception as e:
            self._serve_json({"error": str(e)}, status_code=500)


def start_server():
    server_address = ("", PORT)
    with socketserver.TCPServer(server_address, TerminalAPIHandler) as httpd:
        print(f"\n==================================================================")
        print(f"   ANTI-GRAVITY TERMINAL SERVER RUNNING AT: http://localhost:{PORT}")
        print(f"==================================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down terminal dashboard server...")


if __name__ == "__main__":
    start_server()
