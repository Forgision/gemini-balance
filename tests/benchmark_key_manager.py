import time
import pandas as pd

# Replicating the logic from app/service/key/key_manager.py

class KeyManagerLogic:
    def __init__(self, num_models=50, keys_per_model=100):
        # Setup similar to KeyManager
        self.rate_limit_models = [f"gemini-{i}.0-flash" for i in range(num_models)]
        # KeyManager sorts this in init
        self.rate_limit_models = sorted(self.rate_limit_models, key=len, reverse=True)

        # Create DataFrame
        data = []
        for m in self.rate_limit_models:
            for k in range(keys_per_model):
                data.append({
                    "model_name": m,
                    "is_vertex_key": False,
                    "api_key": f"key_{m}_{k}",
                    "val": 1
                })

        self.df = pd.DataFrame(data)
        self.df.set_index(["model_name", "is_vertex_key", "api_key"], inplace=True)

    def _model_normalization_original(self, model_name: str) -> str:
        # Original logic: sorts every time
        for prefix in sorted(self.rate_limit_models, key=len, reverse=True):
            if model_name.startswith(prefix):
                return prefix
        return model_name

    def _model_normalization_optimized(self, model_name: str) -> str:
        # Optimized: assumes self.rate_limit_models is already sorted
        for prefix in self.rate_limit_models:
            if model_name.startswith(prefix):
                return prefix
        return model_name

    def get_key_logic_original(self, model_name):
        # Original logic with redundant check
        # Note: The original implementation contained both this explicit check and the try/except block.
        if model_name not in self.df.index.get_level_values("model_name"):
            return "fallback"

        try:
            _ = self.df.xs(model_name, level="model_name", drop_level=False)
            return "found"
        except KeyError:
            return "fallback"

    def get_key_logic_optimized(self, model_name):
        # Optimized: try/except only
        try:
            _ = self.df.xs(model_name, level="model_name", drop_level=False)
            return "found"
        except KeyError:
            return "fallback"

def run_benchmark():
    # Setup
    num_models = 20
    keys_per_model = 200
    print(f"Setting up DataFrame with {num_models} models and {keys_per_model} keys each...")
    km = KeyManagerLogic(num_models=num_models, keys_per_model=keys_per_model)
    print(f"Total rows: {len(km.df)}")

    iterations = 50000
    model_to_test = "gemini-5.0-flash"

    print(f"\nRunning {iterations} iterations for each test...")

    # 1. Benchmark _model_normalization
    print("\n--- _model_normalization ---")
    start = time.time()
    for _ in range(iterations):
        km._model_normalization_original(model_to_test)
    end = time.time()
    orig_norm_time = end - start
    print(f"Original: {orig_norm_time:.4f}s")

    start = time.time()
    for _ in range(iterations):
        km._model_normalization_optimized(model_to_test)
    end = time.time()
    opt_norm_time = end - start
    print(f"Optimized: {opt_norm_time:.4f}s")
    print(f"Improvement: {orig_norm_time / opt_norm_time:.2f}x")

    # 2. Benchmark get_key logic (redundant check)
    print("\n--- get_key logic (existence check) ---")

    # Case 1: Model exists
    print(f"Case: Model '{model_to_test}' exists")
    start = time.time()
    for _ in range(iterations):
        km.get_key_logic_original(model_to_test)
    end = time.time()
    orig_get_time = end - start
    print(f"Original: {orig_get_time:.4f}s")

    start = time.time()
    for _ in range(iterations):
        km.get_key_logic_optimized(model_to_test)
    end = time.time()
    opt_get_time = end - start
    print(f"Optimized: {opt_get_time:.4f}s")
    print(f"Improvement: {orig_get_time / opt_get_time:.2f}x")

    # Case 2: Model does not exist
    missing_model = "model_missing"
    print(f"\nCase: Model '{missing_model}' does not exist")
    start = time.time()
    for _ in range(iterations):
        km.get_key_logic_original(missing_model)
    end = time.time()
    orig_miss_time = end - start
    print(f"Original: {orig_miss_time:.4f}s")

    start = time.time()
    for _ in range(iterations):
        km.get_key_logic_optimized(missing_model)
    end = time.time()
    opt_miss_time = end - start
    print(f"Optimized: {opt_miss_time:.4f}s")
    print(f"Improvement: {orig_miss_time / opt_miss_time:.2f}x")

if __name__ == "__main__":
    run_benchmark()
