import timeit
import sys
import os
from unittest.mock import Mock

# Add project root to path
sys.path.append(os.getcwd())

from app.service.key.key_manager import KeyManager

def benchmark():
    # Setup
    rate_limit_data = {
        "gemini-pro": {},
        "gemini-2.0-flash-exp": {},
        "gemini-2.5-pro": {},
        "gemini-2.5-flash": {},
        "claude-3-opus": {},
        "claude-3-sonnet": {},
        "gpt-4-turbo": {},
        "gpt-3.5-turbo": {},
        "very-long-model-name-prefix-1": {},
        "very-long-model-name-prefix-2": {},
        "short": {},
    }

    # Add more dummy models to make the list larger (simulate 50 models)
    for i in range(40):
        rate_limit_data[f"extra-model-{i}"] = {}

    km = KeyManager(
        api_keys=["k1"],
        vertex_api_keys=[],
        async_session_maker=Mock(),
        rate_limit_data=rate_limit_data
    )

    # Check if rate_limit_models is sorted
    assert km.rate_limit_models == sorted(list(rate_limit_data.keys()), key=len, reverse=True)
    print("Verification passed: rate_limit_models is sorted.")

    model_names = [
        "gemini-pro",
        "gemini-2.5-flash-search",
        "gpt-4-turbo-preview",
        "unknown-model",
        "extra-model-20",
        "very-long-model-name-prefix-1-special"
    ]

    def run_normalization():
        for name in model_names:
            km._model_normalization(name)

    # Run benchmark
    iterations = 10000
    time = timeit.timeit(run_normalization, number=iterations)

    print(f"Time for {iterations} iterations: {time:.4f} seconds")
    print(f"Average time per call: {time / (iterations * len(model_names)) * 1e6:.2f} microseconds")

if __name__ == "__main__":
    benchmark()
