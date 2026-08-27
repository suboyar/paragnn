#!/usr/bin/env python3

from math import ceil, sqrt, floor

# This script is based on "Analytical Modeling Is Enough for High-Performance BLIS"

SP = 4
DP = 8

partiation_to_target_label = {
    # Intel x86-64
    "xeonmaxq": "TARGET_CPU_XEONMAX9480", # Sapphire Rapids (Golden Cove), AVX-512
    "habanaq":  "TARGET_CPU_XEON8360Y",   # Ice Lake (Sunny Cove), AVX-512
    "h200q":    "TARGET_CPU_XEON6960P",   # Granite Rapids (Redwood Cove), AVX-512
    # AMD x86-64
    "defq":     "TARGET_CPU_EPYC7601",    # Naples (Zen), AVX2
    "rome16q":  "TARGET_CPU_EPYC7302P",   # Rome (Zen 2), AVX2
    "milanq":   "TARGET_CPU_EPYC7763",    # Milan B (Zen 3), AVX2
    "fpgaq":    "TARGET_CPU_EPYC7413",    # Milan A (Zen 3), AVX2
    "genoaxq":  "TARGET_CPU_EPYC9684X",   # Genoa (Zen 4), AVX-512
    # ARM
    "armq":     "TARGET_CPU_THUNDERX2",   # ThunderX2, ASIMD
    "huaq":     "TARGET_CPU_KUNPENG920",  # Kunpeng 920, ASIMD
    "gh200q":   "TARGET_CPU_NEOVERSEV2",  # Grace (Neoverse V2), SVE2
}

micro_tile_params = {
    # Intel x86-64
    "xeonmaxq": {"vlen": 512, "Lvfma": 4, "Nvfma": 2, "regs": 32},
    "habanaq":  {"vlen": 512, "Lvfma": 4, "Nvfma": 2, "regs": 32},
    "h200q":    {"vlen": 512, "Lvfma": 4, "Nvfma": 2, "regs": 32},
    # AMD x86-64
    "defq":     {"vlen": 256, "Lvfma": 5, "Nvfma": 2, "regs": 16},
    "rome16q":  {"vlen": 256, "Lvfma": 5, "Nvfma": 2, "regs": 16},
    "fpgaq":    {"vlen": 256, "Lvfma": 4, "Nvfma": 2, "regs": 16},
    "milanq":   {"vlen": 256, "Lvfma": 4, "Nvfma": 2, "regs": 16},
    "genoaxq":  {"vlen": 512, "Lvfma": 4, "Nvfma": 2, "regs": 32},
    # ARM
    "armq":     {"vlen": 128, "Lvfma": 6, "Nvfma": 2, "regs": 32},
    "huaq":     {"vlen": 128, "Lvfma": 5, "Nvfma": 2, "regs": 32},
    "gh200q":   {"vlen": 128, "Lvfma": 4, "Nvfma": 4, "regs": 32},
}

cache_params = {
    # [Size in bytes, Associativity (Ways)]
    # Intel x86-64
    "xeonmaxq": {"L1": [48*1024, 12], "L2": [2048*1024, 16],  "L3": [112.5*1024*1024, 15]},
    "habanaq":  {"L1": [48*1024, 12], "L2": [1280*1024, 20],  "L3": [54*1024*1024, 12]},
    "h200q":    {"L1": [48*1024, 12], "L2": [2048*1024, 16],  "L3": [432*1024*1024, 16]},
    # AMD X86-64
    "defq":     {"L1": [32*1024, 8], "L2": [512*1024, 8],  "L3": [8*1024*1024, 16]},
    "rome16q":  {"L1": [32*1024, 8], "L2": [512*1024, 8],  "L3": [16*1024*1024, 16]},
    "fpgaq":    {"L1": [32*1024, 8], "L2": [512*1024, 8],  "L3": [32*1024*1024, 16]},
    "milanq":   {"L1": [32*1024, 8], "L2": [512*1024, 8],  "L3": [32*1024*1024, 16]},
    "genoaxq":  {"L1": [32*1024, 8], "L2": [1024*1024, 8],  "L3": [98304*1024, 16]},
    # ARM
    "armq":     {"L1": [32*1024, 8], "L2": [256*1024, 8],  "L3": [32*1024*1024, 32]},
    "huaq":     {"L1": [64*1024, 4], "L2": [512*1024, 8],  "L3": [32*1024*1024, 15]},
    "gh200q":   {"L1": [64*1024, 4], "L2": [1024*1024, 8],  "L3": [114*1024*1024, 12]},
}

def get_micro_tiles(Nvec, Lvfma, Nvfma, regs):
    # Standard BLIS analytical model (Optimal for hiding latency on e.g. AVX2)
    if regs <= 16:
        product = Nvec * Lvfma * Nvfma
        nr = ceil(sqrt(product) / Nvec) * Nvec
        mr = ceil(product / nr)
        return mr, nr

    # Has abundant registers (AVX-512, 32+), maximize compute-to-load ratio
    best_mr = 0
    best_nv = 0
    max_intensity = 0.0

    for nv in range(1, regs):
        for mr in range(2, regs, 2):
            used_regs = (mr * nv) + nv + 1
            if used_regs <= regs:
                ops = mr * nv
                loads = mr + nv
                intensity = ops / loads
                if (intensity > max_intensity) or (intensity == max_intensity and ops > best_mr * best_nv):
                    max_intensity = intensity
                    best_mr = mr
                    best_nv = nv

    return best_mr, best_nv * Nvec

def get_unroll_factors(mr, nv, regs, reg_usage_func):
    factors = {}
    f = 1
    while (usage := reg_usage_func(mr, nv, f)) < regs:
        factors[f] = {"usage": usage, "percentage": usage/regs}
        f *= 2
    return factors

def get_kc_block(mr, nr, Sdata, Sl1, Wl1):
    # Special case for 2-way associative caches
    if Wl1 == 2:
        kc = (Sl1 / Wl1) / (nr * Sdata)
        return floor(kc)

    # Calculate cache lines per set dedicated to the streaming Bp panel
    C_Bp = floor((Wl1 - 1) / (1 + (mr / nr)))

    # Calculate optimal KC
    cache_way_size = Sl1 / Wl1
    kc = (C_Bp * cache_way_size) / (nr * Sdata)
    return floor(kc)

# outer_tn_v3
def outer_tn_v3_reg_usage_broadcast_a(mr, nv, k_unroll):
    """Kernel: Loads whole Cr, broadcasts A, needs 'nv' vectors for B."""
    regs_for_c = mr * nv
    return int(regs_for_c + ((nv + 1) * k_unroll))

first = True
for name, params in micro_tile_params.items():
    Sdata = SP
    Nvec = floor(params["vlen"] / (Sdata * 8))
    mr, nr = get_micro_tiles(Nvec, params["Lvfma"], params["Nvfma"], params["regs"])
    nv = nr / Nvec
    factors = get_unroll_factors(mr, nv, params["regs"], outer_tn_v3_reg_usage_broadcast_a)
    k_unroll = max(factors.keys()) if len(factors) > 0 else 1
    c_params = cache_params[name]
    kc = get_kc_block(mr, nr, Sdata, c_params["L1"][0], c_params["L1"][1])

    if first:
        print("#if ", end="")
    else:
        print("#elif ", end="")

    print(f"    defined({partiation_to_target_label[name]}) /* {name} */")
    print(f"    #define MR {mr}")
    print(f"    #define NR {nr}")
    print(f"    #define KC {kc}")
    print(f"    #define K_UNROLL {k_unroll}")

    first = False
print("#else")
print("    #error \"Target CPU not supported or defined.\"")
print("#endif")

print("\n-----------------------------------------------")
print(" Register usage for different k unroll factors ")
print("-----------------------------------------------")

for name, params in micro_tile_params.items():
    Nvec = floor(params["vlen"] / 32)
    # mr, nr = get_micro_tiles(Nvec, params["Lvfma"], params["Nvfma"])
    mr, nr = get_micro_tiles(Nvec, params["Lvfma"], params["Nvfma"], params["regs"])
    nv = nr / Nvec
    factors = get_unroll_factors(mr, nv, params["regs"], outer_tn_v3_reg_usage_broadcast_a)
    print(f"{name}:")
    for factor,v in factors.items():
        print(f"  k-unroll = {factor} {{register usage: {v['usage']} ({v['percentage']:.1%})}}")
    print()
