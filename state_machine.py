"""
SYSTEM ARCHITECTURE SPECIFICATION: AI-DRIVEN XAUUSD SCALPER (ANTI-GRAVITY)
State Machine Specification: IDLE -> SCANNING -> SIGNAL_FIRED -> VALIDATING -> EXECUTING -> MANAGING -> FLAT
"""

from enum import Enum
from typing import Callable, Dict, List, Optional
try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("antigravity")
    logger.success = logger.info


class BotState(str, Enum):
    IDLE = "IDLE"
    SCANNING = "SCANNING"
    SIGNAL_FIRED = "SIGNAL_FIRED"
    VALIDATING = "VALIDATING"
    EXECUTING = "EXECUTING"
    MANAGING = "MANAGING"
    FLAT = "FLAT"


class StateMachine:
    """
    Finite State Machine governing the scalper lifecycle.
    Enforces valid deterministic transitions and lifecycle callbacks.
    """
    def __init__(self, initial_state: BotState = BotState.IDLE):
        self._current_state = initial_state
        self._listeners: List[Callable[[BotState, BotState], None]] = []
        
        # Valid transition graph
        self._valid_transitions: Dict[BotState, List[BotState]] = {
            BotState.IDLE: [BotState.SCANNING, BotState.FLAT],
            BotState.SCANNING: [BotState.SIGNAL_FIRED, BotState.SCANNING, BotState.FLAT, BotState.IDLE],
            BotState.SIGNAL_FIRED: [BotState.VALIDATING, BotState.SCANNING, BotState.FLAT],
            BotState.VALIDATING: [BotState.EXECUTING, BotState.SCANNING, BotState.FLAT],
            BotState.EXECUTING: [BotState.MANAGING, BotState.SCANNING, BotState.FLAT],
            BotState.MANAGING: [BotState.MANAGING, BotState.FLAT, BotState.SCANNING],
            BotState.FLAT: [BotState.IDLE, BotState.SCANNING]
        }

    @property
    def current_state(self) -> BotState:
        return self._current_state

    def add_transition_listener(self, callback: Callable[[BotState, BotState], None]):
        self._listeners.append(callback)

    def transition_to(self, new_state: BotState, reason: str = "") -> bool:
        """Transitions state if allowed by transition rules"""
        if new_state == self._current_state:
            return True

        allowed = self._valid_transitions.get(self._current_state, [])
        if new_state not in allowed:
            logger.warning(
                f"[STATE MACHINE] Invalid transition attempted: {self._current_state.value} -> {new_state.value}. Allowed: {[s.value for s in allowed]}"
            )
            return False

        old_state = self._current_state
        self._current_state = new_state
        logger.info(f"[STATE MACHINE] Transition: [{old_state.value}] -> [{new_state.value}] (Reason: {reason})")

        for listener in self._listeners:
            try:
                listener(old_state, new_state)
            except Exception as e:
                logger.error(f"[STATE MACHINE] Error in state listener: {e}")

        return True
