"""Chat-window nudge polling for ACP turns.

Plugin build 1.221.4 never injects nudges into tool results for the generic
ACP client (only CopilotClient wires AgentNudgeService), so pending human
nudges accumulate until AUTO_SEND flushes them as plain text at turn end.
Hermes-side compensation: read the plugin conversation store over PsiBridge
HTTP MCP (query_conversation_history) and steer newly seen [nudge] lines into
the live agent mid-turn.
"""

import json
import logging
import re
import threading
import urllib.request

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.5
_PSIBRIDGE_URL = "http://127.0.0.1:8642/mcp"
_HISTORY_WINDOW = 3
_NUDGE_LINE = re.compile(r"^>>> \[nudge\] (.+)$")
_TURN_LINE = re.compile(r"^=== \[turn_id:([0-9a-f-]+)\]")


class _PsiBridgeClient:
    def __init__(self):
        self._sid = None
        self._req_id = 0
        self._lock = threading.Lock()

    def _post(self, payload):
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._sid:
            headers["Mcp-Session-Id"] = self._sid
        req = urllib.request.Request(
            _PSIBRIDGE_URL, data=json.dumps(payload).encode(),
            headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode(errors="replace")
            return resp.status, resp.headers.get("Mcp-Session-Id"), body

    def _initialize(self):
        self._req_id += 1
        _, sid, _ = self._post({
            "jsonrpc": "2.0", "id": self._req_id, "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "hermes-nudge-poller", "version": "0.1"},
            },
        })
        self._sid = sid
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call_tool(self, name, arguments):
        with self._lock:
            if not self._sid:
                self._initialize()
            self._req_id += 1
            try:
                _, _, body = self._post({
                    "jsonrpc": "2.0", "id": self._req_id,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                })
            except Exception:
                self._sid = None
                self._initialize()
                _, _, body = self._post({
                    "jsonrpc": "2.0", "id": self._req_id,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                })
        if body and not body.lstrip().startswith("{"):
            for line in body.splitlines():
                if line.startswith("data:"):
                    body = line[5:].strip()
                    break
        data = json.loads(body)
        return "\n".join(
            c.get("text", "") for c in data.get("result", {}).get("content", [])
            if c.get("type") == "text"
        )


def _extract_nudges(history_text):
    found = []
    turn_id = None
    for line in history_text.splitlines():
        m = _TURN_LINE.match(line)
        if m:
            turn_id = m.group(1)
            continue
        m = _NUDGE_LINE.match(line)
        if m and turn_id:
            found.append((turn_id, m.group(1).strip()))
    return found


class NudgePoller:
    def __init__(self, session_id: str, agent, stop_event: threading.Event):
        self._session_id = session_id
        self._agent = agent
        self._stop_event = stop_event
        self._thread = None
        self._seen = set()
        self._client = _PsiBridgeClient()
        self._warned = False

    def start(self):
        self._thread = threading.Thread(
            target=self._run, name=f"nudge-poll-{self._session_id[:8]}", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _seed_seen(self):
        try:
            text = self._client.call_tool(
                "query_conversation_history", {"last_n": _HISTORY_WINDOW}
            )
            for turn_id, nudge in _extract_nudges(text):
                self._seen.add((turn_id, nudge))
        except Exception:
            logger.debug("Nudge poller seed failed", exc_info=True)

    def _run(self):
        logger.info("Nudge poller started for session %s", self._session_id)
        self._seed_seen()
        while not self._stop_event.wait(_POLL_INTERVAL_S):
            try:
                text = self._client.call_tool(
                    "query_conversation_history", {"last_n": _HISTORY_WINDOW}
                )
            except Exception as exc:
                if not self._warned:
                    self._warned = True
                    logger.warning(
                        "Nudge poller: PsiBridge unreachable (%s); nudges paused", exc
                    )
                continue
            self._warned = False
            for turn_id, nudge in _extract_nudges(text):
                if (turn_id, nudge) in self._seen:
                    continue
                self._seen.add((turn_id, nudge))
                logger.info(
                    "Nudge poller: captured nudge for session %s: %r",
                    self._session_id, nudge[:120],
                )
                self._deliver(nudge)
        logger.info("Nudge poller stopped for session %s", self._session_id)

    def _deliver(self, nudge):
        try:
            if hasattr(self._agent, "redirect") and self._agent.redirect(nudge):
                return
        except Exception:
            logger.debug("Nudge redirect failed", exc_info=True)
        try:
            if hasattr(self._agent, "steer"):
                self._agent.steer(nudge)
        except Exception:
            logger.debug("Nudge steer failed", exc_info=True)


def start_nudge_poller(session_id: str, agent) -> NudgePoller:
    stop_event = threading.Event()
    poller = NudgePoller(session_id, agent, stop_event)
    poller.start()
    return poller
