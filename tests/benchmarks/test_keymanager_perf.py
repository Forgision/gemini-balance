import asyncio
import pandas as pd
import time
import pytest
from app.service.key.key_manager import KeyManager
from app.utils.read_write_lock import ReadWriteLock
from unittest.mock import MagicMock

# Mock KeyManager to isolate update_usage
class MockKeyManager(KeyManager):
    def __init__(self):
        self.lock = ReadWriteLock()
        self.now = lambda: pd.Timestamp.now()
        self.now_minute = lambda: self.now().replace(second=0, microsecond=0)
        self.now_day = lambda: self.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.df = pd.DataFrame()
        self.is_ready = True
        self._required_db_commit = False
        self.rate_limit_models = ["gemini-pro"]
        self.rate_limit_data = {"gemini-pro": {"RPM": 60, "TPM": 1000000, "RPD": 1500}}

    async def _on_update_usage(self):
        # We want to measure the impact of removing this too
        await self._set_available_usage()
        await self._set_exhausted_flags()
        self._required_db_commit = True

    async def _set_available_usage(self):
        self.df["rpm_left"] = (self.df["max_rpm"] - self.df["rpm"]).clip(lower=0)
        self.df["tpm_left"] = (self.df["max_tpm"] - self.df["tpm"]).clip(lower=0)
        self.df["rpd_left"] = (self.df["max_rpd"] - self.df["rpd"]).clip(lower=0)

    async def _set_exhausted_flags(self):
        flags = (
            (self.df["rpm"] >= self.df["max_rpm"])
            | (self.df["tpm"] >= self.df["max_tpm"])
            | (self.df["rpd"] >= self.df["max_rpd"])
        )
        if "is_exhausted" in self.df.columns:
            self.df["is_exhausted"] = self.df["is_exhausted"] | flags
        else:
            self.df["is_exhausted"] = flags

    def _model_normalization(self, model_name):
        return model_name

async def benchmark():
    km = MockKeyManager()

    # Setup DF with some rows
    rows = []
    models = ["gemini-pro"]
    keys = [f"key-{i}" for i in range(100)]
    for m in models:
        for k in keys:
             rows.append({
                "api_key": k,
                "model_name": m,
                "rpm": 0,
                "max_rpm": 60,
                "tpm": 0,
                "max_tpm": 1000000,
                "rpd": 0,
                "max_rpd": 1500,
                "minute_reset_time": km.now_minute(),
                "day_reset_time": km.now_day(),
                "last_used": km.now(),
                "is_vertex_key": False,
                "is_active": True,
                "is_exhausted": False,
                "total_token_count": 0
            })

    km.df = pd.DataFrame(rows)
    km.df.set_index(["model_name", "is_vertex_key", "api_key"], inplace=True)
    await km._ensure_numeric_columns()

    start_time = time.time()
    iterations = 1000
    for i in range(iterations):
        await km.update_usage(
            model_name="gemini-pro",
            key_value="key-0",
            is_vertex_key=False,
            tokens_used=10
        )
    end_time = time.time()

    print(f"Total time for {iterations} updates: {end_time - start_time:.4f}s")
    print(f"Time per update: {(end_time - start_time) / iterations * 1000:.4f}ms")

if __name__ == "__main__":
    asyncio.run(benchmark())
