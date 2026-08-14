"""Read the names in a Home Assistant install over the websocket API.

Read-only and recognition-only: this fetches the names a speaker can actually
say -- exposed entities, the areas and floors they live in -- and never calls a
service or handles an intent. The result is a
[RecognitionContext](vocabulary.py) that biases speech-to-text.

Only *conversation-exposed* entities are collected. Every entity in the house
would put hundreds of names the speaker cannot mean into a prompt with room for
a few dozen, crowding out the ones they can.

Requires the ``hass`` extra (aiohttp).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Coroutine, Dict, List, Set
from urllib.parse import urlparse, urlunparse

import aiohttp

from .const import HASS_API_URL
from .vocabulary import RecognitionContext, clean_names

_LOGGER = logging.getLogger(__name__)

Command = Callable[[Dict[str, Any]], Coroutine[Any, Any, Dict[str, Any]]]


class HomeAssistantError(Exception):
    """Home Assistant refused a request or the connection failed."""


class HomeAssistant:
    """Read-only client for the names speech-to-text biases toward."""

    def __init__(
        self,
        token: str,
        api_url: str = HASS_API_URL,
        timeout: float = 10.0,
    ) -> None:
        self.token = token
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

        parsed = urlparse(self.api_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

        scheme = "wss" if parsed.scheme == "https" else "ws"
        self.websocket_api_url = urlunparse(
            parsed._replace(
                scheme=scheme,
                path=f"{parsed.path}/websocket",
                params="",
                query="",
                fragment="",
            )
        )

    async def get_context(self) -> RecognitionContext:
        """Fetch the current names. Raises HomeAssistantError on failure."""
        current_id = 0

        def next_id() -> int:
            nonlocal current_id
            current_id += 1
            return current_id

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(
                    self.websocket_api_url, max_msg_size=0
                ) as websocket:

                    async def command(payload: Dict[str, Any]) -> Dict[str, Any]:
                        """One request, one reply. We subscribe to nothing, so
                        nothing else can arrive in between."""
                        await websocket.send_json({"id": next_id(), **payload})
                        msg = await websocket.receive_json()
                        if not msg.get("success"):
                            raise HomeAssistantError(f"{payload['type']} failed: {msg}")

                        return msg

                    await self._authenticate(websocket)
                    return await self._load(command)
        except HomeAssistantError:
            raise
        except Exception as exc:
            raise HomeAssistantError(f"Failed to load names: {exc}") from exc

    async def _authenticate(self, websocket: Any) -> None:
        msg = await websocket.receive_json()
        if msg.get("type") != "auth_required":
            raise HomeAssistantError(f"Expected auth_required, got {msg}")

        await websocket.send_json({"type": "auth", "access_token": self.token})
        msg = await websocket.receive_json()
        if msg.get("type") != "auth_ok":
            raise HomeAssistantError(f"Authentication failed: {msg}")

    async def _load(self, command: Command) -> RecognitionContext:
        # Entities the user exposed to conversation. Nothing else is a possible
        # target of a voice command.
        msg = await command({"type": "homeassistant/expose_entity/list"})
        exposed: Set[str] = {
            entity_id
            for entity_id, info in (msg["result"]["exposed_entities"] or {}).items()
            if info.get("conversation")
        }

        # The displayed (friendly) name is what Home Assistant itself matches and
        # what a user reading their dashboard will say. The registry's
        # name/original_name may have had the device prefix stripped, leaving
        # "Blinds" where the entity shows as "Bedroom Blinds"; biasing toward
        # that fragment would put a bare cover-class word into the prompt.
        friendly: Dict[str, str] = {}
        msg = await command({"type": "get_states"})
        for state in msg["result"]:
            entity_id = state["entity_id"]
            if entity_id not in exposed:
                continue

            attributes = state.get("attributes") or {}
            name = " ".join((attributes.get("friendly_name") or "").split())
            if name:
                friendly[entity_id] = name

        # Aliases only come with the extended registry entries, so exposed
        # entities are fetched a second time to get them. An alias is what a
        # speaker actually says when the integration's own name is not it.
        entries: Dict[str, Dict[str, Any]] = {}
        if exposed:
            msg = await command(
                {
                    "type": "config/entity_registry/get_entries",
                    "entity_ids": sorted(exposed),
                }
            )
            entries = {k: v for k, v in msg["result"].items() if v}

        entity_names: List[str] = []
        alias_names: List[str] = []
        unnamed: List[str] = []
        for entity_id in sorted(exposed):
            entry = entries.get(entity_id) or {}
            if entry.get("disabled_by") is not None:
                continue

            name = friendly.get(entity_id) or entry.get("name") or ""
            if not name:
                name = entry.get("original_name") or ""

            if name:
                entity_names.append(name)
            else:
                unnamed.append(entity_id)

            alias_names.extend(entry.get("aliases") or [])

        area_names: List[str] = []
        msg = await command({"type": "config/area_registry/list"})
        for area in msg["result"]:
            area_names.append(area.get("name") or "")
            area_names.extend(area.get("aliases") or [])

        floor_names: List[str] = []
        msg = await command({"type": "config/floor_registry/list"})
        for floor in msg["result"]:
            floor_names.append(floor.get("name") or "")
            floor_names.extend(floor.get("aliases") or [])

        context = RecognitionContext(
            areas=clean_names(area_names),
            floors=clean_names(floor_names),
            entities=clean_names(entity_names),
            aliases=clean_names(alias_names),
        )
        # One line per fetch: this runs once per utterance, so anything per-entity
        # here would bury the rest of the log.
        _LOGGER.debug(
            "Loaded names from Home Assistant: %s areas, %s floors, "
            "%s entities, %s aliases%s",
            len(context.areas),
            len(context.floors),
            len(context.entities),
            len(context.aliases),
            (
                f" (skipped {len(unnamed)} unnamed: {', '.join(unnamed)})"
                if unnamed
                else ""
            ),
        )
        return context


__all__ = [
    "HASS_API_URL",
    "HomeAssistant",
    "HomeAssistantError",
]
