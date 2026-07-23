from math import ceil, sqrt, floor

# This script is based on "Analytical Modeling Is Enough for High-Performance BLIS"

micro_tile_params = {
    # Intel x86-64
    "xeonmaxq": {"vlen": 512, "Lvfma": 4, "Nvfma": 2, "regs": 32},   # Sapphire Rapids (Golden Cove), AVX-512
    "habanaq":  {"vlen": 512, "Lvfma": 4, "Nvfma": 2, "regs": 32},   # Ice Lake (Sunny Cove), AVX-512
    "h200q":    {"vlen": 512, "Lvfma": 4, "Nvfma": 2, "regs": 32},   # Granite Rapids (Redwood Cove), AVX-512
    # AMD x86-64
    "defq":     {"vlen": 256, "Lvfma": 5, "Nvfma": 2, "regs": 16},   # Naples (Zen), AVX2
    "rome16q":  {"vlen": 256, "Lvfma": 5, "Nvfma": 2, "regs": 16},   # Rome (Zen 2), AVX2
    "fpgaq":    {"vlen": 256, "Lvfma": 4, "Nvfma": 2, "regs": 16},   # Milan A (Zen 3), AVX2
    "milanq":   {"vlen": 256, "Lvfma": 4, "Nvfma": 2, "regs": 16},   # Milan B (Zen 3), AVX2
    "genoaxq":  {"vlen": 512, "Lvfma": 4, "Nvfma": 2, "regs": 32},   # Genoa (Zen 4), AVX-512
    # ARM
    "armq":     {"vlen": 128, "Lvfma": 6, "Nvfma": 2, "regs": 32},   # ThunderX2, ASIMD
    "huaq":     {"vlen": 128, "Lvfma": 5, "Nvfma": 2, "regs": 32},   # Kunpeng 920, ASIMD
    "gh200q":   {"vlen": 128, "Lvfma": 4, "Nvfma": 4, "regs": 32},   # Grace (Neoverse V2), SVE2
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

def get_micro_tiles(Nvec, Lvfma, Nvfma):
    product = Nvec * Lvfma * Nvfma
    nr = ceil(sqrt(product) / Nvec) * Nvec
    mr = ceil(product / nr)
    return mr, nr

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

def get_nc_block(kc, nr, Sdata, Sl2, Wl2):
    # Reserve 2 ways in L2 cache for Ap and C_work, dedicate the rest to Bp
    usable_L2 = Sl2 * ((Wl2 - 2) / Wl2)
    nc = usable_L2 / (kc * Sdata)

    # Round down to the nearest multiple of nr
    return floor(nc / nr) * nr

# outer_tn_v5
def outer_tn_v5_reg_usage_broadcast_a(mr, nv, k_unroll):
    """Kernel: Loads whole Cr, broadcasts A, needs 'nv' vectors for B."""
    regs_for_c = mr * nv
    return int(regs_for_c + ((nv + 1) * k_unroll))

for name, params in micro_tile_params.items():
    Sdata = 4 # SP, 8 for DP
    Nvec = floor(params["vlen"] / (Sdata * 8))
    mr, nr = get_micro_tiles(Nvec, params["Lvfma"], params["Nvfma"])
    nv = nr / Nvec

    factors = get_unroll_factors(mr, nv, params["regs"], outer_tn_v5_reg_usage_broadcast_a)
    k_unroll = max(factors.keys())

    c_params = cache_params[name]

    kc = get_kc_block(mr, nr, Sdata, c_params["L1"][0], c_params["L1"][1])
    nc = get_nc_block(kc, nr, Sdata, c_params["L2"][0], c_params["L2"][1])

    print(f"\n{name}:")
    print(f"  Microkernel: -DMR={mr} -DNR={nr}")
    print(f"  v5 Flags:    -KC={kc} -K_UNROLL={k_unroll}")
    print(f"    #define KC {kc}")
    print(f"    #define K_UNROLL {k_unroll}")
    print(f"  v6 Flags:    -DKC={kc} -DNC={nc} -K_UNROLL={k_unroll}")
    print(f"    #define KC {kc}")
    print(f"    #define NC {nc}")
    print(f"    #define K_UNROLL {k_unroll}")

print("\n-----------------------------------------------")
print(" Register usage for different k unroll factors ")
print("-----------------------------------------------")

for name, params in micro_tile_params.items():
    Nvec = floor(params["vlen"] / 32)
    mr, nr = get_micro_tiles(Nvec, params["Lvfma"], params["Nvfma"])
    nv = nr / Nvec
    factors = get_unroll_factors(mr, nv, params["regs"], outer_tn_v5_reg_usage_broadcast_a)
    print(f"{name}:")
    for factor,v in factors.items():
        print(f"  k-unroll = {factor} {{register usage: {v['usage']} ({v['percentage']:.1%})}}")
    print()
