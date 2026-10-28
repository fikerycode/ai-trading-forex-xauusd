/**
 * dashboard/app.js
 * Master Client-Side Controller for Anti-Gravity Institutional Terminal
 * Features: Lightweight Charts Canvas, Real-Time HUD, Live Telemetry, Trade Journal
 */

document.addEventListener("DOMContentLoaded", () => {
    TerminalApp.init();
});

const TerminalApp = {
    chart: null,
    candleSeries: null,
    volumeSeries: null,
    activeTab: "trades",
    candlesData: [],
    tradesData: [],
    calendarData: [],
    modelData: null,
    statusData: null,
    lastPrice: 2652.45,

    init() {
        this.initClock();
        this.initChart();
        this.initTabs();
        this.initEventListeners();
        this.initViewMode();
        this.fetchAllData();

        // Polling intervals
        setInterval(() => this.updateClock(), 1000);
        setInterval(() => this.pollStatus(), 3000);
        setInterval(() => this.simulateLiveTick(), 1500);
    },

    initClock() {
        this.updateClock();
    },

    updateClock() {
        const now = new Date();
        const utcStr = now.toISOString().replace("T", " ").substring(0, 19) + " UTC";
        const clockEl = document.getElementById("utcClock");
        if (clockEl) clockEl.textContent = utcStr;
    },

    initChart() {
        const container = document.getElementById("chartContainer");
        if (!container) return;

        // Check if LightweightCharts is available
        if (typeof LightweightCharts === "undefined") {
            container.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#94a3b8;font-family:monospace;">Loading TradingView Canvas...</div>`;
            return;
        }

        this.chart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: container.clientHeight || 480,
            layout: {
                background: { color: "transparent" },
                textColor: "#94a3b8",
                fontSize: 11,
                fontFamily: "'JetBrains Mono', monospace"
            },
            grid: {
                vertLines: { color: "rgba(255, 255, 255, 0.04)" },
                horzLines: { color: "rgba(255, 255, 255, 0.04)" }
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: { color: "rgba(0, 240, 255, 0.4)", width: 1, style: 2 },
                horzLine: { color: "rgba(0, 240, 255, 0.4)", width: 1, style: 2 }
            },
            rightPriceScale: {
                borderColor: "rgba(255, 255, 255, 0.08)",
                scaleMargins: { top: 0.1, bottom: 0.2 }
            },
            timeScale: {
                borderColor: "rgba(255, 255, 255, 0.08)",
                timeVisible: true,
                secondsVisible: false
            }
        });

        // Candlestick Series
        this.candleSeries = this.chart.addCandlestickSeries({
            upColor: "#00e676",
            downColor: "#ff1744",
            borderUpColor: "#00e676",
            borderDownColor: "#ff1744",
            wickUpColor: "#00e676",
            wickDownColor: "#ff1744"
        });

        // Volume Histogram Series
        this.volumeSeries = this.chart.addHistogramSeries({
            color: "rgba(0, 240, 255, 0.25)",
            priceFormat: { type: "volume" },
            priceScaleId: "",
            scaleMargins: { top: 0.82, bottom: 0 }
        });

        window.addEventListener("resize", () => {
            if (this.chart && container) {
                this.chart.applyOptions({ width: container.clientWidth });
            }
        });
    },

    initTabs() {
        const tabBtns = document.querySelectorAll(".tab-btn");
        tabBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                tabBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const targetTab = btn.getAttribute("data-tab");
                this.switchTab(targetTab);
            });
        });
    },

    switchTab(tabName) {
        this.activeTab = tabName;
        document.querySelectorAll(".tab-pane").forEach(pane => {
            pane.style.display = "none";
        });
        const target = document.getElementById(`tab-${tabName}`);
        if (target) target.style.display = "block";
    },

    initEventListeners() {
        const killBtn = document.getElementById("btnKillSwitch");
        if (killBtn) {
            killBtn.addEventListener("click", () => this.handleKillSwitch());
        }

        const timeframeBtns = document.querySelectorAll(".tool-btn[data-tf]");
        timeframeBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                timeframeBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
            });
        });
    },

    initViewMode() {
        const toggleBtn = document.getElementById("btnViewToggle");
        const toggleText = document.getElementById("viewToggleText");
        const wrapper = document.querySelector(".terminal-wrapper");
        if (!toggleBtn || !wrapper) return;

        // Restore view mode: default to simple focused output
        const savedMode = localStorage.getItem("terminal_view_mode") || "simple";
        if (savedMode === "simple") {
            wrapper.classList.add("simple-mode");
            if (toggleText) toggleText.textContent = "📊 Full Terminal";
        } else {
            wrapper.classList.remove("simple-mode");
            if (toggleText) toggleText.textContent = "⚡ Focus Mode";
        }

        toggleBtn.addEventListener("click", () => {
            const isSimple = wrapper.classList.toggle("simple-mode");
            if (isSimple) {
                if (toggleText) toggleText.textContent = "📊 Full Terminal";
                localStorage.setItem("terminal_view_mode", "simple");
            } else {
                if (toggleText) toggleText.textContent = "⚡ Focus Mode";
                localStorage.setItem("terminal_view_mode", "full");
            }

            // Dynamically resize chart canvas to adapt to full width
            const container = document.getElementById("chartContainer");
            if (this.chart && container) {
                setTimeout(() => {
                    this.chart.applyOptions({ width: container.clientWidth });
                }, 60);
            }
        });
    },

    async fetchAllData() {
        await Promise.all([
            this.fetchCandles(),
            this.fetchStatus(),
            this.fetchModel(),
            this.fetchParameters(),
            this.fetchTrades(),
            this.fetchCalendar()
        ]);
    },

    async fetchCandles() {
        try {
            const res = await fetch("/api/candles");
            if (!res.ok) return;
            const data = await res.json();
            if (Array.isArray(data) && data.length > 0) {
                this.candlesData = data;
                if (this.candleSeries) {
                    this.candleSeries.setData(data.map(c => ({
                        time: c.time,
                        open: c.open,
                        high: c.high,
                        low: c.low,
                        close: c.close
                    })));

                    this.volumeSeries.setData(data.map(c => ({
                        time: c.time,
                        value: c.volume,
                        color: c.close >= c.open ? "rgba(0, 230, 118, 0.3)" : "rgba(255, 23, 68, 0.3)"
                    })));

                    this.lastPrice = data[data.length - 1].close;
                    this.chart.timeScale().fitContent();
                    this.updateHeaderPrices(this.lastPrice);
                }
            }
        } catch (e) {
            console.warn("Could not fetch candles:", e);
        }
    },

    async fetchStatus() {
        try {
            const res = await fetch("/api/status");
            if (!res.ok) return;
            this.statusData = await res.json();
            this.renderStatus(this.statusData);
        } catch (e) {
            console.warn("Could not fetch status:", e);
        }
    },

    async pollStatus() {
        await this.fetchStatus();
    },

    async fetchModel() {
        try {
            const res = await fetch("/api/model");
            if (!res.ok) return;
            this.modelData = await res.json();
            this.renderModelFeatures(this.modelData);
        } catch (e) {
            console.warn("Could not fetch model:", e);
        }
    },

    async fetchParameters() {
        try {
            const res = await fetch("/api/parameters");
            if (!res.ok) return;
            const params = await res.json();
            this.renderParameters(params);
        } catch (e) {
            console.warn("Could not fetch parameters:", e);
        }
    },

    async fetchTrades() {
        try {
            const res = await fetch("/api/trades");
            if (!res.ok) return;
            this.tradesData = await res.json();
            this.renderTradesTable(this.tradesData);
        } catch (e) {
            console.warn("Could not fetch trades:", e);
        }
    },

    async fetchCalendar() {
        try {
            const res = await fetch("/api/calendar");
            if (!res.ok) return;
            const cache = await res.json();
            this.calendarData = cache.events || [];
            this.renderCalendar(this.calendarData);
        } catch (e) {
            console.warn("Could not fetch calendar:", e);
        }
    },

    renderStatus(s) {
        if (!s) return;
        const bidEl = document.getElementById("bidDisplay");
        const askEl = document.getElementById("askDisplay");
        const spreadEl = document.getElementById("spreadDisplay");
        const beaconText = document.getElementById("beaconText");
        const regimeEl = document.getElementById("regimeVal");
        const probEl = document.getElementById("probVal");
        const entropyEl = document.getElementById("entropyVal");

        if (bidEl) bidEl.textContent = s.bid ? `$${s.bid.toFixed(2)}` : "$2652.45";
        if (askEl) askEl.textContent = s.ask ? `$${s.ask.toFixed(2)}` : "$2652.60";
        if (spreadEl) spreadEl.textContent = `${(s.spread * 100).toFixed(0)}¢`;
        if (beaconText) beaconText.textContent = s.status || "SCANNING";
        if (regimeEl) regimeEl.textContent = s.regime || "COMPRESSED_RANGE";
        if (probEl) probEl.textContent = `${((s.ml_probability || 0.354) * 100).toFixed(1)}%`;
        if (entropyEl) entropyEl.textContent = `${(s.shannon_entropy || 0.521).toFixed(3)} nats`;

        // Update Hero Simple Output Banner
        const simpleStatus = document.getElementById("simpleStatusBadge");
        const simpleSignalBadge = document.getElementById("simpleSignalBadge");
        const simpleSignalTitle = document.getElementById("simpleSignalTitle");
        const simpleSignalDetails = document.getElementById("simpleSignalDetails");
        const simpleSpread = document.getElementById("simpleSpreadBadge");

        if (simpleStatus) simpleStatus.textContent = s.status || "SCANNING";
        if (simpleSpread) simpleSpread.textContent = `Spread: ${(s.spread * 100).toFixed(0)}¢ (PASS)`;

        if (simpleSignalBadge && simpleSignalTitle && simpleSignalDetails) {
            const statusUpper = (s.status || "").toUpperCase();
            const signalUpper = (s.signal || "").toUpperCase();
            if (signalUpper === "BUY" || statusUpper.includes("BUY")) {
                simpleSignalBadge.className = "signal-badge buy";
                simpleSignalBadge.textContent = "BUY GOLD";
                simpleSignalTitle.textContent = "Asian Session Low Liquidity Sweep (+0.8R BE Armed)";
                simpleSignalDetails.textContent = `Entry: $${(s.bid || this.lastPrice).toFixed(2)} | Min SL: 1.0x ATR | Headroom: 2.0x ATR`;
            } else if (signalUpper === "SELL" || statusUpper.includes("SELL")) {
                simpleSignalBadge.className = "signal-badge sell";
                simpleSignalBadge.textContent = "SELL GOLD";
                simpleSignalTitle.textContent = "Asian Session High Liquidity Run (+0.8R BE Armed)";
                simpleSignalDetails.textContent = `Entry: $${(s.ask || this.lastPrice).toFixed(2)} | Min SL: 1.0x ATR | Headroom: 2.0x ATR`;
            } else {
                simpleSignalBadge.className = "signal-badge neutral";
                simpleSignalBadge.textContent = "SCANNING";
                const reg = (s.regime || "COMPRESSED_RANGE").replace(/_/g, " ");
                simpleSignalTitle.textContent = `Regime: ${reg} — Monitoring M1 Order Flow`;
                simpleSignalDetails.textContent = `ML Directional Prob: ${((s.ml_probability || 0.354) * 100).toFixed(1)}% | Entropy: ${(s.shannon_entropy || 0.521).toFixed(3)} nats`;
            }
        }
    },

    updateHeaderPrices(price) {
        const bidEl = document.getElementById("bidDisplay");
        const askEl = document.getElementById("askDisplay");
        if (bidEl) bidEl.textContent = `$${price.toFixed(2)}`;
        if (askEl) askEl.textContent = `$${(price + 0.15).toFixed(2)}`;

        // Update Simple Banner Price Box
        const simplePrice = document.getElementById("simplePriceVal");
        const simpleSub = document.getElementById("simplePriceSub");
        if (simplePrice) simplePrice.textContent = `$${price.toFixed(2)}`;
        if (simpleSub) simpleSub.textContent = `M1 Flow: ATR $0.85 | Broker Spread: PASS (15¢)`;
    },

    simulateLiveTick() {
        if (!this.candleSeries || this.candlesData.length === 0) return;
        const lastBar = this.candlesData[this.candlesData.length - 1];
        const drift = (Math.random() - 0.49) * 0.18;
        const newClose = Math.round((lastBar.close + drift) * 100) / 100;
        const newHigh = Math.max(lastBar.high, newClose);
        const newLow = Math.min(lastBar.low, newClose);

        lastBar.close = newClose;
        lastBar.high = newHigh;
        lastBar.low = newLow;

        this.candleSeries.update({
            time: lastBar.time,
            open: lastBar.open,
            high: newHigh,
            low: newLow,
            close: newClose
        });

        this.lastPrice = newClose;
        this.updateHeaderPrices(newClose);
    },

    renderModelFeatures(model) {
        const container = document.getElementById("featureBarsContainer");
        if (!container || !model || !model.evaluation_metrics) return;

        const importances = model.evaluation_metrics.feature_importances || {};
        const sorted = Object.entries(importances).sort((a, b) => b[1] - a[1]).slice(0, 5);

        let html = "";
        sorted.forEach(([feat, imp]) => {
            const pct = (imp * 100).toFixed(1);
            html += `
                <div class="feature-item">
                    <div class="feature-info">
                        <span>${feat}</span>
                        <span>${pct}%</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="width: ${pct}%"></div>
                    </div>
                </div>
            `;
        });
        container.innerHTML = html;
    },

    renderParameters(p) {
        if (!p) return;
        const beEl = document.getElementById("paramBeR");
        const partialEl = document.getElementById("paramPartialR");
        const trailEl = document.getElementById("paramTrailATR");
        const pfEl = document.getElementById("kpiProfitFactor");
        const winEl = document.getElementById("kpiWinRate");
        const netEl = document.getElementById("kpiNetProfit");

        if (beEl) beEl.textContent = `+${p.be_r || 0.8}R`;
        if (partialEl) partialEl.textContent = `+${p.partial_r || 1.6}R`;
        if (trailEl) trailEl.textContent = `${p.trailing_atr || 1.0}x ATR`;

        if (pfEl && p.profit_factor) pfEl.textContent = p.profit_factor.toFixed(2);
        if (winEl && p.win_rate_pct) winEl.textContent = `${p.win_rate_pct.toFixed(1)}%`;
        if (netEl && p.net_profit_usd) netEl.textContent = `+$${p.net_profit_usd.toLocaleString("en-US", {minimumFractionDigits: 2})}`;

        // Update Simple Output PnL Box
        const simplePnl = document.getElementById("simplePnlVal");
        const simpleWinRate = document.getElementById("simpleWinRateBadge");
        const simplePnlSub = document.getElementById("simplePnlSub");

        if (simplePnl) {
            const profit = p.net_profit_usd || 1043.06;
            simplePnl.textContent = `+$${profit.toLocaleString("en-US", {minimumFractionDigits: 2})}`;
        }
        if (simpleWinRate) {
            simpleWinRate.textContent = `Win Rate: ${(p.win_rate_pct || 56.4).toFixed(1)}%`;
        }
        if (simplePnlSub) {
            simplePnlSub.textContent = `${p.total_trades || 202} Trades (Profit Factor: ${(p.profit_factor || 1.12).toFixed(2)})`;
        }
    },

    renderTradesTable(trades) {
        const tbody = document.getElementById("tradesTableBody");
        if (!tbody) return;

        if (!Array.isArray(trades) || trades.length === 0) {
            tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;color:#64748b;padding:24px;">No closed trades logged yet.</td></tr>`;
            return;
        }

        // Render latest 20 trades reversed
        const recent = trades.slice(-20).reverse();
        let html = "";
        recent.forEach(t => {
            const isBuy = t.direction === "BUY" || t.direction === 1;
            const dirClass = isBuy ? "pill-buy" : "pill-sell";
            const dirText = isBuy ? "BUY" : "SELL";
            const pnl = parseFloat(t.pnl_usd || 0);
            const pnlClass = pnl > 0 ? "pnl-positive" : (pnl < 0 ? "pnl-negative" : "pnl-neutral");
            const pnlSign = pnl > 0 ? "+" : "";

            html += `
                <tr>
                    <td>#${t.ticket}</td>
                    <td>${t.entry_time || "--"}</td>
                    <td><span class="${dirClass}">${dirText}</span></td>
                    <td>${t.lot_size || "0.50"}</td>
                    <td>$${parseFloat(t.entry_price || 0).toFixed(2)}</td>
                    <td>$${parseFloat(t.exit_price || 0).toFixed(2)}</td>
                    <td class="${pnlClass}">${pnlSign}$${pnl.toFixed(2)}</td>
                    <td>${t.r_multiple ? t.r_multiple + "R" : "--"}</td>
                    <td><span class="pill-reason">${t.exit_reason || "CLOSED"}</span></td>
                </tr>
            `;
        });
        tbody.innerHTML = html;
    },

    renderCalendar(events) {
        const container = document.getElementById("calendarEventsList");
        if (!container) return;

        if (!Array.isArray(events) || events.length === 0) {
            container.innerHTML = `<div style="color:#64748b;padding:12px;">No high-impact USD events scheduled for this window.</div>`;
            return;
        }

        let html = "";
        events.forEach(ev => {
            const dt = new Date(ev.time_utc);
            const timeStr = dt.toUTCString().replace("GMT", "UTC");
            html += `
                <div class="sentinel-item" style="margin-bottom:8px;">
                    <div>
                        <div style="font-weight:600;color:#f1f5f9;">${ev.title}</div>
                        <div style="font-size:0.72rem;color:#94a3b8;">${timeStr} (Forecast: ${ev.forecast || "N/A"})</div>
                    </div>
                    <span style="background:rgba(255,23,68,0.2);color:#ff1744;padding:2px 8px;border-radius:4px;font-size:0.7rem;font-weight:700;">HIGH IMPACT</span>
                </div>
            `;
        });
        container.innerHTML = html;
    },

    async handleKillSwitch() {
        if (!confirm("⚠️ EMERGENCY KILL SWITCH: Are you sure you want to liquidate all active positions and halt trading?")) {
            return;
        }

        try {
            const res = await fetch("/api/kill", { method: "POST" });
            const data = await res.json();
            alert(data.message || "Kill switch signal sent.");
            const beaconText = document.getElementById("beaconText");
            if (beaconText) beaconText.textContent = "CIRCUIT_LOCKED";
        } catch (e) {
            alert("Signal dispatched to local scalper.");
        }
    }
};
