"""
SQLite database manager for caching model scans and storing patches.

Uses aiosqlite for async access.  Provides a repository-pattern
interface so the storage backend can be swapped to Redis later.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite
from loguru import logger


class DatabaseManager:
    """Async SQLite wrapper with table auto-creation."""

    def __init__(self, db_path: str = "steerops.db") -> None:
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    # ── Lifecycle ─────────────────────────────────────────────
    async def initialize(self) -> None:
        """Open the DB connection and create tables if needed."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        # Enable WAL mode for concurrent read/write performance
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._create_tables()
        logger.info(f"Database ready at {self.db_path} (WAL mode)")

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def _create_tables(self) -> None:
        assert self._db is not None
        await self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS model_profiles (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name  TEXT    NOT NULL UNIQUE,
                architecture TEXT   NOT NULL DEFAULT '',
                num_layers  INTEGER NOT NULL DEFAULT 0,
                hidden_dim  INTEGER NOT NULL DEFAULT 0,
                scan_hash   TEXT    NOT NULL DEFAULT '',
                scanned_at  TEXT    NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS layer_mappings (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                model_profile_id INTEGER NOT NULL,
                layer_index     INTEGER NOT NULL,
                category        TEXT    NOT NULL DEFAULT 'unknown',
                confidence      REAL    NOT NULL DEFAULT 0.0,
                behavioral_role TEXT    NOT NULL DEFAULT '',
                weight_stats    TEXT    NOT NULL DEFAULT '{}',
                description     TEXT    NOT NULL DEFAULT '',
                FOREIGN KEY (model_profile_id)
                    REFERENCES model_profiles(id) ON DELETE CASCADE,
                UNIQUE(model_profile_id, layer_index)
            );

            CREATE TABLE IF NOT EXISTS patches (
                id              TEXT    PRIMARY KEY,
                name            TEXT    NOT NULL,
                model_name      TEXT    NOT NULL DEFAULT '',
                description     TEXT    NOT NULL DEFAULT '',
                patch_data      TEXT    NOT NULL DEFAULT '{}',
                created_at      TEXT    NOT NULL DEFAULT '',
                file_size_kb    REAL    NOT NULL DEFAULT 0.0
            );
            """
        )
        await self._db.commit()

    # ╔══════════════════════════════════════════════════════════╗
    # ║             MODEL PROFILES (scan cache)                 ║
    # ╚══════════════════════════════════════════════════════════╝

    async def get_model_profile(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Return cached profile or None."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM model_profiles WHERE model_name = ?", (model_name,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def save_model_profile(
        self,
        model_name: str,
        architecture: str,
        num_layers: int,
        hidden_dim: int,
        scan_hash: str,
    ) -> int:
        """Insert or update a model profile. Returns the profile id."""
        assert self._db is not None
        now = datetime.utcnow().isoformat() + "Z"
        await self._db.execute(
            """
            INSERT INTO model_profiles
                (model_name, architecture, num_layers, hidden_dim, scan_hash, scanned_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name) DO UPDATE SET
                architecture = excluded.architecture,
                num_layers   = excluded.num_layers,
                hidden_dim   = excluded.hidden_dim,
                scan_hash    = excluded.scan_hash,
                scanned_at   = excluded.scanned_at
            """,
            (model_name, architecture, num_layers, hidden_dim, scan_hash, now),
        )
        await self._db.commit()
        cursor = await self._db.execute(
            "SELECT id FROM model_profiles WHERE model_name = ?", (model_name,)
        )
        row = await cursor.fetchone()
        return row["id"]  # type: ignore[index]

    # ╔══════════════════════════════════════════════════════════╗
    # ║                  LAYER MAPPINGS                         ║
    # ╚══════════════════════════════════════════════════════════╝

    async def get_layer_mappings(
        self, model_name: str
    ) -> List[Dict[str, Any]]:
        """Fetch all cached layer mappings for a model."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT lm.* FROM layer_mappings lm
            JOIN model_profiles mp ON lm.model_profile_id = mp.id
            WHERE mp.model_name = ?
            ORDER BY lm.layer_index
            """,
            (model_name,),
        )
        rows = await cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["weight_stats"] = json.loads(d.get("weight_stats", "{}"))
            result.append(d)
        return result

    async def save_layer_mappings(
        self,
        model_profile_id: int,
        layers: List[Dict[str, Any]],
    ) -> None:
        """Bulk-insert layer mappings (replaces existing for this profile)."""
        assert self._db is not None
        await self._db.execute(
            "DELETE FROM layer_mappings WHERE model_profile_id = ?",
            (model_profile_id,),
        )
        for layer in layers:
            await self._db.execute(
                """
                INSERT INTO layer_mappings
                    (model_profile_id, layer_index, category, confidence,
                     behavioral_role, weight_stats, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_profile_id,
                    layer["layer_index"],
                    layer["category"],
                    layer["confidence"],
                    layer.get("behavioral_role", ""),
                    json.dumps(layer.get("weight_stats", {})),
                    layer.get("description", ""),
                ),
            )
        await self._db.commit()

    # ╔══════════════════════════════════════════════════════════╗
    # ║                      PATCHES                            ║
    # ╚══════════════════════════════════════════════════════════╝

    async def save_patch(
        self,
        patch_id: str,
        name: str,
        model_name: str,
        description: str,
        patch_data: Dict[str, Any],
    ) -> None:
        assert self._db is not None
        now = datetime.utcnow().isoformat() + "Z"
        data_str = json.dumps(patch_data)
        size_kb = len(data_str.encode()) / 1024
        await self._db.execute(
            """
            INSERT OR REPLACE INTO patches
                (id, name, model_name, description, patch_data, created_at, file_size_kb)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (patch_id, name, model_name, description, data_str, now, round(size_kb, 2)),
        )
        await self._db.commit()

    async def get_patches(self) -> List[Dict[str, Any]]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT id, name, model_name, description, created_at, file_size_kb "
            "FROM patches ORDER BY created_at DESC"
        )
        return [dict(r) for r in await cursor.fetchall()]

    async def get_patch(self, patch_id: str) -> Optional[Dict[str, Any]]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM patches WHERE id = ?", (patch_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        d = dict(row)
        d["patch_data"] = json.loads(d.get("patch_data", "{}"))
        return d
