"""按原顺序生成四份随机表，并核验原始数组与嵌套前缀 hash。"""

import hashlib
import json
import numpy as np
from config import EVIDENCE

COUNTS = tuple(range(16, 129, 16))
DATA_SEED = 202609112301


def array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def random_table(dataset):
    rng = np.random.default_rng(DATA_SEED + 1_000_003 * dataset)
    order = rng.permutation(256).astype(np.int64)
    labels = rng.integers(0, 2, 256, dtype=np.uint8)
    inputs = ((np.arange(256, dtype=np.uint32)[:, None]
               >> np.arange(7, -1, -1)) & 1).astype(np.float32)
    return inputs, labels, order


def verify_data():
    expected = json.loads((EVIDENCE / "datasets.json").read_text(encoding="utf-8"))
    for dataset in range(4):
        x, y, order = random_table(dataset)
        record = expected[str(dataset)]
        assert array_digest(x, y, order) == record["raw_dataset_sha256"]
        for n in COUNTS:
            selected = order[:n]
            assert array_digest(x[selected], y[selected], selected) == record["prefix_sha256"][str(n)]
    return {"tables": 4, "prefixes": 32, "training_started": False, "smc_started": False}


if __name__ == "__main__":
    print(json.dumps(verify_data(), ensure_ascii=False))

