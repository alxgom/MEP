"""
Dataset Generator for Steiner Tournament
========================================
Provides standardized terminal sets for benchmarking.
"""

import json
import os
import random
from typing import List, Tuple, Dict, Any

def load_presets() -> List[Dict[str, Any]]:
    path = os.path.join(os.path.dirname(__file__), "..", "02-steiner-playground", "steiner_presets.json")
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data.get("presets", [])
    except Exception:
        return []

def generate_random_cluster(n: int, seed: int = 42) -> List[Tuple[float, float]]:
    rng = random.Random(seed)
    return [(rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(n)]

def get_benchmark_suites() -> Dict[str, List[Dict[str, Any]]]:
    presets = load_presets()
    
    suites = {
        "Classical": [],
        "Random_Small": [],
        "Random_Medium": [],
        "Random_Large": []
    }
    
    for p in presets:
        if "terminals" in p:
            suites["Classical"].append({
                "name": p["name"],
                "terminals": [tuple(pt) for pt in p["terminals"]]
            })
            
    for n in [5, 10]:
        suites["Random_Small"].append({
            "name": f"Random_{n}",
            "terminals": generate_random_cluster(n, seed=n)
        })
        
    for n in [20, 30]:
        suites["Random_Medium"].append({
            "name": f"Random_{n}",
            "terminals": generate_random_cluster(n, seed=n)
        })

    for n in [50]:
        suites["Random_Large"].append({
            "name": f"Random_{n}",
            "terminals": generate_random_cluster(n, seed=n)
        })
        
    return suites
