"""
news_filter.py
Forex Factory Economic Calendar Ingestion & Real-Time Macro Sentinel.

Connects to the official Forex Factory JSON feed (Fair Economy Media)
to halt trading 15 minutes before and 20 minutes after high-impact US macro events:
- Non-Farm Payrolls (NFP)
- Consumer Price Index (CPI) & PPI
- Federal Open Market Committee (FOMC) Rate Decisions & Press Conferences
- Gross Domestic Product (GDP)
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import json
import os
from typing import Dict, List, Optional, Tuple

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("antigravity.news")
    logger.success = logger.info

import aiohttp
import requests
from dateutil import parser as date_parser

from config import SystemConfig, CONFIG

# Official Fair Economy Media / Forex Factory weekly JSON endpoint
FOREX_FACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CACHE_FILE = "forex_factory_cache.json"


@dataclass(frozen=True)
class MacroEvent:
    title: str
    country: str
    impact: str              # "High", "Medium", "Low", "Holiday"
    time_utc: datetime
    forecast: str = ""
    previous: str = ""


class NewsFilter:
    """
    Forex Factory Economic Calendar Filter.
    - Synchronizes weekly calendar from Forex Factory JSON endpoint.
    - Filters for High-Impact USD releases.
    - Enforces hard lockout interval:
      * 15 minutes before release time.
      * 20 minutes after release time.
    - Caches responses locally to survive connection dropouts.
    """
    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        lockout_before_min: int = 15,
        lockout_after_min: int = 20,
        monitored_currency: str = "USD",
        cache_ttl_hours: int = 4
    ):
        self.config = config or CONFIG
        self.lockout_before = timedelta(minutes=lockout_before_min)
        self.lockout_after = timedelta(minutes=lockout_after_min)
        self.monitored_currency = monitored_currency.upper()
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

        self._high_impact_events: List[MacroEvent] = []
        self._last_sync_time: Optional[datetime] = None
        self._is_syncing: bool = False

        # Load cached events upon initialization
        self._load_cache()

    def _load_cache(self):
        """Loads cached events from local storage if available."""
        if not os.path.exists(CACHE_FILE):
            return

        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            saved_events = []
            for item in data.get("events", []):
                t_utc = datetime.fromisoformat(item["time_utc"])
                saved_events.append(
                    MacroEvent(
                        title=item["title"],
                        country=item["country"],
                        impact=item["impact"],
                        time_utc=t_utc,
                        forecast=item.get("forecast", ""),
                        previous=item.get("previous", "")
                    )
                )

            self._high_impact_events = saved_events
            last_sync_str = data.get("last_sync")
            if last_sync_str:
                self._last_sync_time = datetime.fromisoformat(last_sync_str)

            logger.info(f"[NEWS FILTER] Loaded {len(self._high_impact_events)} high-impact events from cache.")
        except Exception as e:
            logger.warning(f"[NEWS FILTER] Failed loading calendar cache: {e}")

    def _save_cache(self):
        """Saves parsed high-impact events to local cache file."""
        try:
            data = {
                "last_sync": self._last_sync_time.isoformat() if self._last_sync_time else None,
                "events": [
                    {
                        "title": ev.title,
                        "country": ev.country,
                        "impact": ev.impact,
                        "time_utc": ev.time_utc.isoformat(),
                        "forecast": ev.forecast,
                        "previous": ev.previous
                    }
                    for ev in self._high_impact_events
                ]
            }
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"[NEWS FILTER] Error persisting calendar cache: {e}")

    def parse_calendar_payload(self, raw_events: List[Dict]) -> List[MacroEvent]:
        """
        Extracts High-Impact USD macroeconomic releases.
        Standardizes timestamps to UTC.
        """
        high_impact = []
        for item in raw_events:
            country = str(item.get("country", "")).strip().upper()
            impact = str(item.get("impact", "")).strip()

            # Filter for High-Impact USD
            if country != self.monitored_currency or impact.lower() != "high":
                continue

            date_str = item.get("date")
            if not date_str:
                continue

            try:
                # Parse ISO date with timezone offset
                dt = date_parser.isoparse(date_str)
                dt_utc = dt.astimezone(timezone.utc)
            except Exception:
                try:
                    dt = date_parser.parse(date_str)
                    dt_utc = dt.replace(tzinfo=timezone.utc)
                except Exception as err:
                    logger.debug(f"[NEWS FILTER] Skipping unparseable date {date_str}: {err}")
                    continue

            high_impact.append(
                MacroEvent(
                    title=item.get("title", "High Impact USD Release"),
                    country=country,
                    impact="High",
                    time_utc=dt_utc,
                    forecast=str(item.get("forecast", "")),
                    previous=str(item.get("previous", ""))
                )
            )

        # Sort chronologically
        high_impact.sort(key=lambda x: x.time_utc)
        return high_impact

    async def sync_calendar_async(self, force: bool = False) -> bool:
        """
        Asynchronously fetches and refreshes the Forex Factory calendar.
        Respects cache TTL to avoid redundant network queries.
        """
        now = datetime.now(timezone.utc)
        if not force and self._last_sync_time and (now - self._last_sync_time) < self.cache_ttl:
            return True

        if self._is_syncing:
            return False

        self._is_syncing = True
        logger.info(f"[NEWS FILTER] Fetching live economic calendar from Forex Factory...")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }

        try:
            try:
                async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get(FOREX_FACTORY_URL) as response:
                        if response.status == 200:
                            raw_data = await response.json(content_type=None)
                            if isinstance(raw_data, list):
                                self._high_impact_events = self.parse_calendar_payload(raw_data)
                                self._last_sync_time = now
                                self._save_cache()
                                logger.success(
                                    f"[NEWS FILTER] Calendar updated: {len(self._high_impact_events)} high-impact USD events scheduled this week."
                                )
                                return True
                        else:
                            logger.warning(f"[NEWS FILTER] Forex Factory returned HTTP {response.status}")
            except Exception as e:
                logger.warning(f"[NEWS FILTER] Async fetch failed: {e}. Retrying via fallback...")

            # Synchronous fallback via requests
            return self.sync_calendar_sync()
        finally:
            self._is_syncing = False

    def sync_calendar_sync(self) -> bool:
        """Synchronous fallback fetch for Forex Factory calendar."""
        now = datetime.now(timezone.utc)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }
        try:
            resp = requests.get(FOREX_FACTORY_URL, headers=headers, timeout=8)
            if resp.status_code == 200:
                raw_data = resp.json()
                if isinstance(raw_data, list):
                    self._high_impact_events = self.parse_calendar_payload(raw_data)
                    self._last_sync_time = now
                    self._save_cache()
                    logger.success(f"[NEWS FILTER] Sync successful: {len(self._high_impact_events)} high-impact USD events active.")
                    return True
        except Exception as e:
            logger.error(f"[NEWS FILTER] Synchronous calendar fetch failed: {e}")
        finally:
            self._is_syncing = False

        return len(self._high_impact_events) > 0

    def is_news_locked(self, current_time: Optional[datetime] = None) -> Tuple[bool, Optional[str], Optional[MacroEvent]]:
        """
        Evaluates whether trading is currently prohibited due to macro releases.
        Lockout window: [Event_Time - 15m, Event_Time + 20m].
        
        Returns:
            (is_locked: bool, reason: Optional[str], active_event: Optional[MacroEvent])
        """
        now = current_time or datetime.now(timezone.utc)

        for event in self._high_impact_events:
            lock_start = event.time_utc - self.lockout_before
            lock_end = event.time_utc + self.lockout_after

            if lock_start <= now <= lock_end:
                minutes_delta = (event.time_utc - now).total_seconds() / 60.0
                if minutes_delta >= 0:
                    timing_desc = f"in {minutes_delta:.1f} minutes"
                else:
                    timing_desc = f"{abs(minutes_delta):.1f} minutes ago"

                reason = (
                    f"FOREX FACTORY NEWS LOCKOUT: '{event.title}' scheduled at "
                    f"{event.time_utc.strftime('%H:%M UTC')} ({timing_desc}). "
                    f"Trading suppressed until {lock_end.strftime('%H:%M UTC')}."
                )
                return True, reason, event

        return False, None, None

    def get_next_high_impact_event(self, current_time: Optional[datetime] = None) -> Optional[Tuple[MacroEvent, float]]:
        """
        Finds the closest upcoming high-impact event and minutes remaining.
        Returns: (event, minutes_until_event) or None.
        """
        now = current_time or datetime.now(timezone.utc)
        for event in self._high_impact_events:
            if event.time_utc > now:
                minutes_left = (event.time_utc - now).total_seconds() / 60.0
                return event, round(minutes_left, 1)
        return None

    def add_manual_event(self, title: str, event_time_utc: datetime):
        """Allows programmatic addition of bespoke releases or testing events."""
        event = MacroEvent(
            title=title,
            country=self.monitored_currency,
            impact="High",
            time_utc=event_time_utc
        )
        self._high_impact_events.append(event)
        self._high_impact_events.sort(key=lambda x: x.time_utc)
        logger.info(f"[NEWS FILTER] Added manual event: {title} at {event_time_utc.strftime('%Y-%m-%d %H:%M UTC')}")


# Singleton instance for direct import
NEWS_FILTER = NewsFilter()
