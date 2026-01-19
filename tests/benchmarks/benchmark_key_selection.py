import timeit
import pandas as pd
import numpy as np

def setup_dataframe(n_keys=1000):
    index = pd.MultiIndex.from_product(
        [["gemini-pro"], [False], [f"key_{i}" for i in range(n_keys)]],
        names=["model_name", "is_vertex_key", "api_key"]
    )
    data = {
        "tpm_left": np.random.randint(0, 1000000, size=n_keys),
        "is_active": [True] * n_keys,
        "is_exhausted": [False] * n_keys,
    }
    return pd.DataFrame(data, index=index)

def method_sort(df):
    return df.sort_values(by="tpm_left", ascending=False).index[0]

def method_idxmax(df):
    return df["tpm_left"].idxmax()

if __name__ == "__main__":
    df = setup_dataframe(1000)

    # Verify correctness
    res_sort = method_sort(df)
    res_idxmax = method_idxmax(df)

    # Note: they might differ if there are duplicates, but both are valid "max"
    val_sort = df.loc[res_sort, "tpm_left"]
    val_idxmax = df.loc[res_idxmax, "tpm_left"]

    assert val_sort == val_idxmax, f"Values differ: {val_sort} != {val_idxmax}"

    print("Correctness verified.")

    # Benchmark
    t_sort = timeit.timeit(lambda: method_sort(df), number=1000)
    t_idxmax = timeit.timeit(lambda: method_idxmax(df), number=1000)

    print(f"Sort time (1000 iter): {t_sort:.4f}s")
    print(f"Idxmax time (1000 iter): {t_idxmax:.4f}s")
    print(f"Speedup: {t_sort / t_idxmax:.2f}x")
