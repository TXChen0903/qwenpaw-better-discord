# -*- coding: utf-8 -*-
"""Enhanced Discord channel for CoPaw.

Adds embed cards, lazy thinking threads, typing indicators, LLM-based
thread renaming, table → image rendering, and dynamic slash-command
registration over the built-in DiscordChannel.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time as _time
from pathlib import Path
from typing import Any

import discord
from agentscope_runtime.engine.schemas.agent_schemas import (
    ContentType,
    ImageContent,
    MessageType,
    TextContent,
)

from copaw.app.channels.base import OnReplySent, ProcessHandler
from copaw.app.channels.discord_.channel import DiscordChannel
from table_renderer import extract_tables, get_renderer

logger = logging.getLogger("copaw.better_discord")

__version__ = "25.0.0"

_CALL_TYPES = frozenset({
    MessageType.FUNCTION_CALL,
    MessageType.PLUGIN_CALL,
    MessageType.MCP_TOOL_CALL,
})
_OUTPUT_TYPES = frozenset({
    MessageType.FUNCTION_CALL_OUTPUT,
    MessageType.PLUGIN_CALL_OUTPUT,
    MessageType.MCP_TOOL_CALL_OUTPUT,
})
_TOOL_TYPES = _CALL_TYPES | _OUTPUT_TYPES

_TITLE_PROMPT = (
    "Based on the following conversation, generate a concise thread title.\n"
    "Rules:\n"
    "1. Detect the primary language used and use the same language.\n"
    "2. CRITICAL for Chinese: Distinguish Traditional (正體中文) from "
    "Simplified (简体中文). Default to Traditional when ambiguous.\n"
    "3. Keep it brief (4–15 characters/words).\n"
    "4. No punctuation, quotes, or explanations — ONLY the title.\n\n"
    "User: {user_text}\n"
    "Assistant: {reply_text}"
)


# ── content helpers ─────────────────────────────────────────────────────────

def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = [str(getattr(i, "text", None) or (i.get("text") if isinstance(i, dict) else None) or "")
                 for i in value]
        return "\n".join(p for p in parts if p)
    t = getattr(value, "text", None)
    if isinstance(value, dict):
        t = t or value.get("text")
    return str(t) if t else str(value)


def _unwrap(data: Any) -> dict | None:
    if isinstance(data, dict):
        inner = data.get("data", data)
        return inner if isinstance(inner, dict) else None
    if isinstance(data, (list, tuple)):
        for item in data:
            d = getattr(item, "data", None)
            if isinstance(d, dict):
                return d
            if isinstance(item, dict) and isinstance(item.get("data"), dict):
                return item["data"]
        if len(data) == 1:
            item = data[0]
            t = getattr(item, "text", None)
            if isinstance(item, dict):
                t = t or item.get("text")
            if t:
                try:
                    parsed = json.loads(str(t))
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    pass
        return None
    try:
        parsed = json.loads(str(data))
        if isinstance(parsed, dict):
            return parsed.get("data", parsed)
        if isinstance(parsed, list) and len(parsed) == 1:
            item = parsed[0]
            if isinstance(item, dict) and "text" in item:
                try:
                    return json.loads(item["text"])
                except (json.JSONDecodeError, TypeError):
                    pass
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _tool_info(content: Any, *, is_output: bool) -> tuple[str | None, Any, str | None]:
    data = _unwrap(content)
    if not data:
        return None, None, None
    name, call_id = data.get("name"), data.get("call_id")
    payload = data.get("output") if (is_output and data.get("output") is not None) else data.get("arguments")
    if name or call_id:
        return name, payload, call_id
    if is_output:
        return None, data, None
    return None, None, None


# ── formatting helpers ──────────────────────────────────────────────────────

def _trunc(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _parse_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        body = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(body).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _to_fields(payload: Any, max_val: int = 900) -> dict[str, str] | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        obj = _parse_json(payload)
        return {"result": payload} if obj is None else _to_fields(obj, max_val)
    if isinstance(payload, (list, tuple)):
        texts = [str(getattr(i, "text", None) or (i.get("text") if isinstance(i, dict) else None) or "")
                 for i in payload]
        texts = [t for t in texts if t]
        if not texts:
            return None
        if len(texts) == 1:
            obj = _parse_json(texts[0])
            return {"result": texts[0]} if obj is None else _to_fields(obj, max_val)
        return {"result": "\n".join(texts)}
    if isinstance(payload, dict):
        if not payload:
            return None
        if len(payload) > 30:
            return {"result": json.dumps(payload, ensure_ascii=False, indent=2)}
        return {
            k: _trunc(
                json.dumps(v, ensure_ascii=False, indent=2) if isinstance(v, (dict, list)) else str(v),
                max_val,
            )
            for k, v in payload.items()
        }
    if isinstance(payload, list):
        return {"result": json.dumps(payload, ensure_ascii=False, indent=2)}
    return {"result": str(payload)}


# ── channel ─────────────────────────────────────────────────────────────────

class BetterDiscordChannel(DiscordChannel):

    channel = "discord"
    _typing_tasks: dict[str, asyncio.Task] = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.color_call: int     = kwargs.pop("embed_color_call", 0x60A5FA)
        self.color_output: int   = kwargs.pop("embed_color_output", 0x34D399)
        self.color_thinking: int = kwargs.pop("embed_color_thinking", 0x818CF8)
        self.thread_name: str    = kwargs.pop("thread_name", "Thinking~")
        self.thread_auto_archive: int = kwargs.pop("thread_auto_archive", 60)
        self.max_field_len: int  = kwargs.pop("max_embed_field_len", 900)
        self.max_output_len: int = kwargs.pop("max_output_text_len", 1800)
        self.max_call_arg_len: int = kwargs.pop("max_call_arg_len", 150)
        self.typing_interval: int  = kwargs.pop("typing_interval", 8)

        super().__init__(*args, **kwargs)

        self._active_thread_id: str | None = None
        self._active_channel_id: str | None = None
        self._active_user_text: str | None = None
        self._active_reply_text: str | None = None
        self._pending_slash: dict[str, discord.Interaction] = {}
        self._slash_started: dict[str, float] = {}

        logger.info("BetterDiscord v%s initialized", __version__)
        self._cleanup_stale_table_temps()

        if self.enabled and self._client is not None:
            self._cmd_tree = discord.app_commands.CommandTree(self._client)

            @self._client.event
            async def on_interaction(interaction: discord.Interaction) -> None:
                await self._cmd_tree.interaction(interaction)

            @self._client.event
            async def on_ready() -> None:
                try:
                    self._register_slash_commands()
                    await self._cmd_tree.sync()
                    logger.info("Slash commands synced (%s)", self._client.user)
                except Exception:
                    logger.error("Failed to sync slash commands", exc_info=True)

    # ── factory methods ─────────────────────────────────────────────────────

    @classmethod
    def from_env(cls, process: ProcessHandler, on_reply_sent: OnReplySent = None) -> "BetterDiscordChannel":
        return cls(
            process=process,
            enabled=os.getenv("DISCORD_CHANNEL_ENABLED", "0") == "1",
            token=os.getenv("DISCORD_BOT_TOKEN", ""),
            http_proxy=os.getenv("DISCORD_HTTP_PROXY", ""),
            http_proxy_auth=os.getenv("DISCORD_HTTP_PROXY_AUTH", ""),
            bot_prefix=os.getenv("DISCORD_BOT_PREFIX", ""),
            on_reply_sent=on_reply_sent,
            dm_policy=os.getenv("DISCORD_DM_POLICY", "open"),
            group_policy=os.getenv("DISCORD_GROUP_POLICY", "open"),
            allow_from=[],
            deny_message=os.getenv("DISCORD_DENY_MESSAGE", ""),
            require_mention=os.getenv("DISCORD_REQUIRE_MENTION", "0") == "1",
            accept_bot_messages=os.getenv("DISCORD_ACCEPT_BOT_MESSAGES", "0") == "1",
        )

    @classmethod
    def from_config(cls, process: ProcessHandler, config: Any, on_reply_sent: OnReplySent = None,
                    show_tool_details: bool = True, filter_tool_messages: bool = False,
                    filter_thinking: bool = False, workspace_dir: Path | None = None,
                    ) -> "BetterDiscordChannel":
        c = config if isinstance(config, dict) else getattr(config, "model_dump", lambda: vars(config))()
        _s = lambda k: (c.get(k) or "").strip()
        return cls(
            process=process,
            enabled=bool(c.get("enabled", False)),
            token=_s("bot_token"),
            http_proxy=_s("http_proxy"),
            http_proxy_auth=_s("http_proxy_auth"),
            bot_prefix=_s("bot_prefix"),
            on_reply_sent=on_reply_sent,
            show_tool_details=show_tool_details,
            filter_tool_messages=filter_tool_messages,
            filter_thinking=filter_thinking,
            dm_policy=c.get("dm_policy") or "open",
            group_policy=c.get("group_policy") or "open",
            allow_from=c.get("allow_from") or [],
            deny_message=_s("deny_message"),
            require_mention=c.get("require_mention", False),
            accept_bot_messages=c.get("accept_bot_messages", False),
        )

    # ── send with slash followup routing ────────────────────────────────────

    async def send(self, to_handle: str, text: str, meta: dict | None = None) -> None:
        cid = (meta or {}).get("channel_id")
        if not cid:
            route = self._route_from_handle(to_handle)
            cid = route.get("channel_id") or None
        interaction = self._pending_slash.get(cid) if cid else None

        if interaction is not None and _time.time() - self._slash_started.get(cid, 0) < 800:
            try:
                for chunk in self._chunk_text(text):
                    await interaction.followup.send(content=chunk)
                self._pending_slash.pop(cid, None)
                self._slash_started.pop(cid, None)
                return
            except Exception:
                logger.debug("followup.send failed, falling back", exc_info=True)
                self._pending_slash.pop(cid, None)
                self._slash_started.pop(cid, None)

        await super().send(to_handle, text, meta)

    # ── table temp cleanup ──────────────────────────────────────────────────

    @staticmethod
    def _cleanup_stale_table_temps(max_age_hours: int = 1) -> None:
        try:
            now = _time.time()
            for d in Path(tempfile.gettempdir()).glob("bd_table_*"):
                if d.is_dir():
                    try:
                        if now - d.stat().st_mtime > max_age_hours * 3600:
                            for f in d.iterdir():
                                f.unlink(missing_ok=True)
                            d.rmdir()
                    except Exception:
                        pass
        except Exception:
            pass

    # ── send_content_parts with table → image ───────────────────────────────

    async def send_content_parts(self, to_handle: str, parts: list[Any],
                                 meta: dict[str, Any] | None = None) -> None:
        text_parts = [p.text for p in parts
                      if getattr(p, "type", None) == ContentType.TEXT and getattr(p, "text", "")]
        if not text_parts:
            return await super().send_content_parts(to_handle, parts, meta)

        all_text = "\n".join(text_parts)
        tables = extract_tables(all_text)
        if not tables:
            return await super().send_content_parts(to_handle, parts, meta)

        renderer = get_renderer()
        rendered_pngs: list[str] = []
        try:
            for table in tables:
                png = await renderer.render([table])
                if png:
                    rendered_pngs.append(png)
        except Exception:
            logger.warning("Table render failed", exc_info=True)
            return await super().send_content_parts(to_handle, parts, meta)
        if not rendered_pngs:
            return await super().send_content_parts(to_handle, parts, meta)

        combined = all_text
        for t in sorted(tables, key=lambda x: -x.start):
            combined = combined[:t.start] + combined[t.end:]
        combined = combined.strip()

        new_parts: list = []
        if combined:
            new_parts.append(TextContent(type=ContentType.TEXT, text=combined))
        for png in rendered_pngs:
            new_parts.append(ImageContent(type=ContentType.IMAGE, image_url=f"file:///{png.replace(os.sep, '/')}"))

        try:
            await super().send_content_parts(to_handle, new_parts, meta)
        finally:
            for png in rendered_pngs:
                renderer.cleanup_temp(png)

    # ── lifecycle hooks ─────────────────────────────────────────────────────

    async def _before_consume_process(self, request: Any) -> None:
        await super()._before_consume_process(request)
        cid = self._get_channel_id(request)
        if cid:
            self._active_channel_id = cid
            self._start_typing(cid)
        for msg in reversed(getattr(request, "input", None) or []):
            if getattr(msg, "role", None) == "user":
                self._active_user_text = _extract_text(getattr(msg, "content", ""))
                break

    async def _on_process_completed(self, request: Any, to_handle: str, send_meta: dict[str, Any]) -> None:
        self._stop_typing(self._active_channel_id)
        if self._active_thread_id and (self._active_user_text or self._active_reply_text):
            asyncio.create_task(self._rename_thread(self._active_thread_id,
                                                     self._active_user_text or "",
                                                     self._active_reply_text or ""))
        self._active_thread_id = self._active_channel_id = None
        self._active_user_text = self._active_reply_text = None
        cid = self._get_channel_id(request)
        if cid:
            self._pending_slash.pop(cid, None)
            self._slash_started.pop(cid, None)
        await super()._on_process_completed(request, to_handle, send_meta)

    async def _on_consume_error(self, request: Any, to_handle: str, err_text: str) -> None:
        self._stop_typing(self._active_channel_id)
        self._active_thread_id = self._active_channel_id = None
        self._active_user_text = self._active_reply_text = None
        cid = self._get_channel_id(request)
        if cid:
            self._pending_slash.pop(cid, None)
            self._slash_started.pop(cid, None)
        await super()._on_consume_error(request, to_handle, err_text)

    # ── typing indicator ────────────────────────────────────────────────────

    async def _typing_loop(self, channel_id: str) -> None:
        route = discord.http.Route("POST", "/channels/{channel_id}/typing", channel_id=int(channel_id))
        try:
            while True:
                try:
                    await self._client.http.request(route)
                except Exception:
                    pass
                await asyncio.sleep(self.typing_interval)
        except asyncio.CancelledError:
            pass

    def _start_typing(self, channel_id: str) -> None:
        if not channel_id:
            return
        existing = self._typing_tasks.get(channel_id)
        if existing and not existing.done():
            return
        self._typing_tasks[channel_id] = asyncio.create_task(self._typing_loop(channel_id))

    def _stop_typing(self, channel_id: str | None) -> None:
        if not channel_id:
            return
        task = self._typing_tasks.pop(channel_id, None)
        if task and not task.done():
            task.cancel()

    # ── event interception ──────────────────────────────────────────────────

    async def on_event_message_completed(self, request: Any, to_handle: str,
                                         event: Any, send_meta: dict[str, Any]) -> None:
        msg_type = getattr(event, "type", None)
        event_content = getattr(event, "content", "")

        if msg_type == "message":
            text = _extract_text(event_content)
            if text:
                self._active_reply_text = text
            return await super().on_event_message_completed(request, to_handle, event, send_meta)

        if msg_type not in _TOOL_TYPES and msg_type != MessageType.REASONING:
            return await super().on_event_message_completed(request, to_handle, event, send_meta)

        thread_id = await self._ensure_thread(request)
        if not thread_id:
            return await super().on_event_message_completed(request, to_handle, event, send_meta)

        embed = self._build_embed(event)
        if embed:
            await self._send_to_thread(thread_id, embed=embed)
        else:
            text = self._build_fallback_text(event)
            if text:
                await self._send_to_thread(thread_id, text=text)
            else:
                await super().on_event_message_completed(request, to_handle, event, send_meta)

    # ── thread management ───────────────────────────────────────────────────

    async def _ensure_thread(self, request: Any) -> str | None:
        if self._active_thread_id:
            return self._active_thread_id
        meta = getattr(request, "channel_meta", {}) or {}
        cid, mid = meta.get("channel_id"), meta.get("message_id")
        if not cid or not mid:
            return None
        try:
            channel = self._client.get_channel(int(cid)) or await self._client.fetch_channel(int(cid))
            msg = await channel.fetch_message(int(mid))
            thread = await msg.create_thread(name=self.thread_name, auto_archive_duration=self.thread_auto_archive)
            self._active_thread_id = str(thread.id)
            return self._active_thread_id
        except Exception:
            logger.warning("Failed to create thread in channel %s", cid, exc_info=True)
            return None

    async def _send_to_thread(self, thread_id: str, *, embed: discord.Embed | None = None,
                              text: str = "") -> None:
        try:
            target = self._client.get_channel(int(thread_id)) or await self._client.fetch_channel(int(thread_id))
            if embed:
                await target.send(embed=embed)
            else:
                for i in range(0, len(text), 2000):
                    await target.send(text[i:i + 2000])
        except Exception:
            logger.warning("Failed to send to thread %s", thread_id, exc_info=True)

    async def _generate_thread_title(self, user_text: str, reply_text: str) -> str | None:
        try:
            from copaw.providers.provider_manager import ProviderManager
            model = ProviderManager.get_active_chat_model()
        except Exception:
            return None
        prompt = _TITLE_PROMPT.format(user_text=user_text[:300], reply_text=reply_text[:400])
        orig = getattr(model, "stream", None)
        try:
            model.stream = False
            resp = await model([{"role": "user", "content": prompt}])
        except Exception:
            return None
        finally:
            if orig is not None:
                model.stream = orig
        content = resp.get("content") if hasattr(resp, "get") else getattr(resp, "content", "")
        if isinstance(content, list):
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
        return re.sub(r"\s+", " ", str(content or "").strip())[:100].strip() or None

    async def _rename_thread(self, thread_id: str, user_text: str, reply_text: str) -> None:
        title = await self._generate_thread_title(user_text, reply_text)
        if not title:
            return
        try:
            await self._client.http.request(
                discord.http.Route("PATCH", "/channels/{channel_id}", channel_id=int(thread_id)),
                json={"name": title},
            )
            logger.info("Thread %s → %s", thread_id, title)
        except Exception:
            logger.warning("Failed to rename thread %s", thread_id, exc_info=True)

    # ── slash command registration ──────────────────────────────────────────

    def _register_slash_commands(self) -> None:
        tree = self._cmd_tree
        tree.clear_commands(guild=None)
        _seen: set[str] = set()

        def _handler(cmd_name: str):
            async def h(interaction: discord.Interaction, args: str = "") -> None:
                text = f"/{cmd_name} {args}".strip() if args else f"/{cmd_name}"
                await self._dispatch_slash(interaction, text)
            h.__name__ = f"_cmd_{cmd_name.replace('-', '_')}"
            return h

        def _register(name: str, description: str) -> None:
            if name in _seen:
                return
            _seen.add(name)
            tree.command(name=name, description=description)(_handler(name))

        # 1. Daemon commands
        try:
            from copaw.app.runner.daemon_commands import DAEMON_SHORT_ALIASES
            for _alias, sub in DAEMON_SHORT_ALIASES.items():
                if sub == "logs":
                    if sub in _seen:
                        continue
                    _seen.add(sub)

                    async def _logs(interaction: discord.Interaction, lines: int = 100) -> None:
                        await self._dispatch_slash(interaction, f"/logs {max(1, min(lines, 2000))}")
                    tree.command(name="logs", description="Tail recent log lines")(_logs)
                else:
                    _register(sub, f"Execute /{sub}")
        except Exception as e:
            logger.warning("Daemon slash commands: %s", e)

        # 2. Control commands (dynamic registry)
        try:
            from copaw.app.runner.control_commands import _COMMAND_REGISTRY
            for cmd_name in _COMMAND_REGISTRY:
                _register(cmd_name.lstrip("/"), f"Execute /{cmd_name.lstrip('/')}")
        except Exception as e:
            logger.warning("Control slash commands: %s", e)

        # 3. Conversation commands
        try:
            from copaw.agents.command_handler import CommandHandler
            for cmd in sorted(CommandHandler.SYSTEM_COMMANDS):
                _register(cmd, f"Execute /{cmd}")
        except Exception as e:
            logger.warning("Conversation slash commands: %s", e)

        logger.info("Slash commands registered (%d): %s", len(_seen), sorted(_seen))

    async def _dispatch_slash(self, interaction: discord.Interaction, cmd_text: str) -> None:
        try:
            await interaction.response.defer(ephemeral=False, thinking=True)
        except Exception:
            pass

        native = {
            "channel_id": self.channel,
            "sender_id": str(interaction.user),
            "content_parts": [TextContent(type=ContentType.TEXT, text=cmd_text)],
            "meta": {
                "user_id": str(interaction.user.id),
                "channel_id": str(interaction.channel_id),
                "guild_id": str(interaction.guild_id) if interaction.guild_id else None,
                "message_id": str(interaction.id),
                "is_dm": interaction.guild_id is None,
                "is_group": interaction.guild_id is not None,
            },
        }

        if self._enqueue:
            cid = native["meta"]["channel_id"]
            self._pending_slash[cid] = interaction
            self._slash_started[cid] = _time.time()
            self._enqueue(native)
        else:
            logger.warning("Slash cmd _enqueue not ready")
            try:
                await interaction.edit_original_response(content="Bot queue not ready, try again shortly.")
            except Exception:
                pass

    # ── embed builders ──────────────────────────────────────────────────────

    def _build_embed(self, event: Any) -> discord.Embed | None:
        msg_type = getattr(event, "type", None)
        content = getattr(event, "content", "")
        if msg_type in _CALL_TYPES:
            return self._build_call_embed(content)
        if msg_type in _OUTPUT_TYPES:
            return self._build_output_embed(content)
        if msg_type == MessageType.REASONING:
            return self._build_thinking_embed(content)
        return None

    def _build_call_embed(self, content: Any) -> discord.Embed:
        name, payload, _ = _tool_info(content, is_output=False)
        embed = discord.Embed(title=f"⚡ {name or 'unknown'}", color=self.color_call)
        if payload is not None:
            for k, v in self._call_fields(payload).items():
                embed.add_field(name=k, value=_trunc(v, self.max_field_len), inline=True)
        return embed

    def _build_output_embed(self, content: Any) -> discord.Embed | None:
        name, payload, _ = _tool_info(content, is_output=True)
        embed = discord.Embed(title=f"✅ {name or 'unknown'}", color=self.color_output)
        fields = _to_fields(payload, max_val=self.max_field_len)
        if not fields:
            return None
        for k, v in fields.items():
            if len(fields) == 1 and len(v) > 100:
                embed.description = _trunc(v, self.max_output_len)
            else:
                embed.add_field(name=k, value=_trunc(v, self.max_field_len), inline=False)
        return embed

    def _build_thinking_embed(self, content: Any) -> discord.Embed | None:
        text = _extract_text(content).strip()
        if not text:
            return None
        orig_len = len(text)
        text = text[:self.max_output_len - 3] + "..." if orig_len > self.max_output_len else text
        text = re.sub(r"\n{3,}", "\n\n", text)
        embed = discord.Embed(title="💭 Thinking", color=self.color_thinking, description=text)
        if orig_len > self.max_output_len:
            embed.set_footer(text=f"truncated ({orig_len} chars)")
        return embed

    def _build_fallback_text(self, event: Any) -> str:
        msg_type = getattr(event, "type", None)
        name, payload, _ = _tool_info(getattr(event, "content", ""), is_output=msg_type in _OUTPUT_TYPES)
        label = name or "unknown"
        text = _extract_text(payload)
        if msg_type in _CALL_TYPES:
            return f"⚡ **{label}**\n{_trunc(text, self.max_call_arg_len)}"
        if msg_type in _OUTPUT_TYPES:
            return f"✅ **{label}**\n{_trunc(text, self.max_output_len)}"
        if msg_type == MessageType.REASONING:
            text = _extract_text(getattr(event, "content", "")).strip()
            return f"💭 **Thinking**\n{_trunc(text, self.max_output_len)}"
        return ""

    def _call_fields(self, payload: Any) -> dict[str, str]:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                return {"args": payload}
        if isinstance(payload, dict) and payload:
            return {k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in payload.items()}
        return {}

    @staticmethod
    def _get_channel_id(request: Any) -> str | None:
        return (getattr(request, "channel_meta", {}) or {}).get("channel_id")
