#!/usr/bin/env python3

from math import ceil, sqrt, floor

# This script is based on "Analytical Modeling Is Enough for High-Performance BLIS"

SP = 4*8
DP = 8*8
Sdata = SP

cpus = {
    # Intel x86-64
    "TARGET_CPU_XEON6960P":    {"vlen": 512, "Lvfma": 4, "Nvfma": 2, "regs": 32, "partition": "h200q"},    # Granite Rapids (Redwood Cove), AVX-512
    "TARGET_CPU_XEONMAX9480":  {"vlen": 512, "Lvfma": 4, "Nvfma": 2, "regs": 32, "partition": "xeonmaxq"}, # Sapphire Rapids (Golden Cove), AVX-512
    "TARGET_CPU_XEON8360Y":    {"vlen": 512, "Lvfma": 4, "Nvfma": 2, "regs": 32, "partition": "habanaq"},  # Ice Lake (Sunny Cove), AVX-512
    # AMD x86-64
    "TARGET_CPU_EPYC9684X":    {"vlen": 512, "Lvfma": 4, "Nvfma": 2, "regs": 32, "partition": "genoaxq"},  # Genoa (Zen 4), AVX-512
    "TARGET_CPU_EPYC7763":     {"vlen": 256, "Lvfma": 4, "Nvfma": 2, "regs": 16, "partition": "milanq"},   # Milan B (Zen 3), AVX2
    "TARGET_CPU_EPYC7413":     {"vlen": 256, "Lvfma": 4, "Nvfma": 2, "regs": 16, "partition": "fpgaq"},    # Milan A (Zen 3), AVX2
    "TARGET_CPU_EPYC7302P":    {"vlen": 256, "Lvfma": 5, "Nvfma": 2, "regs": 16, "partition": "rome16q"},  # Rome (Zen 2), AVX2
    "TARGET_CPU_EPYC7601":     {"vlen": 256, "Lvfma": 5, "Nvfma": 2, "regs": 16, "partition": "defq"},     # Naples (Zen), AVX2
    # ARM
    "TARGET_CPU_NEOVERSEV2":   {"vlen": 128, "Lvfma": 4, "Nvfma": 4, "regs": 32, "partition": "gh200q"},   # Grace (Neoverse V2), SVE2
    "TARGET_CPU_KUNPENG920":   {"vlen": 128, "Lvfma": 5, "Nvfma": 2, "regs": 32, "partition": "huaq"},     # Kunpeng 920, ASIMD
    "TARGET_CPU_THUNDERX2":    {"vlen": 128, "Lvfma": 6, "Nvfma": 2, "regs": 32, "partition": "armq"},     # ThunderX2, ASIMD
}

def optimize_micro_tiles_for_fma_lat(Nvec, Lvfma, Nvfma):
    product = Nvec * Lvfma * Nvfma
    nr = ceil(sqrt(product) / Nvec) * Nvec
    mr = ceil(product / nr)
    return mr, nr

def optimize_micro_tiles_for_compute_to_load(Nvec, regs):
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
    while (usage := reg_usage_func(mr, nv, f)) <= regs:
        factors[f] = {"usage": usage, "percentage": usage/regs}
        f *= 2
    return factors

def outer_tn_v3_reg_usage_broadcast_a(mr, nv, k_unroll):
    """Kernel: Loads whole Cr, broadcasts A, needs 'nv' vectors for B."""
    regs_for_c = mr * nv
    return int(regs_for_c + ((nv + 1) * k_unroll))


# Optimize for fma latency
fma_latency_optimized = {}
for name,specs in cpus.items():
    Nvec = floor(specs["vlen"] / Sdata)
    mr, nr = optimize_micro_tiles_for_fma_lat(Nvec, specs["Lvfma"], specs["Nvfma"])
    nv = nr / Nvec
    factors = get_unroll_factors(mr, nv, specs["regs"], outer_tn_v3_reg_usage_broadcast_a)
    k_unroll = max(factors.keys())
    fma_latency_optimized[name] = {"MR": mr, "NR": nr, "K_UNROLL": k_unroll}


print("\n----------------------------------------------------------------")
print(" Register usage for different k unroll factors for when optimized for fma latency")
print("----------------------------------------------------------------")
for name,params in fma_latency_optimized.items():
    Nvec = floor(cpus[name]["vlen"] / 32)
    mr = params["MR"]
    nr = params["NR"]
    nv = nr / Nvec
    factors = get_unroll_factors(mr, nv, cpus[name]["regs"], outer_tn_v3_reg_usage_broadcast_a)
    print(f"{name}:")
    for factor,v in factors.items():
        print(f"  k-unroll = {factor} {{register usage: {v['usage']} ({v['percentage']:.1%})}}")
    print()
print("----------------------------------------------------------------")


# Optimize for computation-to-load
computation_to_load_optimized = {}
for name,specs in cpus.items():
    Nvec = floor(specs["vlen"] / Sdata)
    mr, nr = optimize_micro_tiles_for_compute_to_load(Nvec, specs["regs"])
    computation_to_load_optimized[name] = {"MR": mr, "NR": nr, "K_UNROLL": 1}

print("#ifndef OUTER_TN_PARAMS_H")
print("#define OUTER_TN_PARAMS_H")
print("#include \"vreg.h\"")
print("")
print("#if defined(FMA_LAT)")
for i, name in enumerate(fma_latency_optimized):
    print(f"{'#if' if i == 0 else '#elif'} defined({name}) /*{cpus[name]['partition']}*/")
    print(f"    #define DEF_KC 1")
    print(f"    #define DEF_MR {fma_latency_optimized[name]['MR']}")
    print(f"    #define DEF_NR {fma_latency_optimized[name]['NR']}")
    print(f"    #define K_UNROLL {fma_latency_optimized[name]['K_UNROLL']}")
print("#else")
print("    #error \"Target CPU not supported or defined.\"")
print("#endif")
print("#else")
for i, name in enumerate(fma_latency_optimized):
    print(f"{'#if' if i == 0 else '#elif'} defined({name}) /*{cpus[name]['partition']}*/")
    print(f"    #define DEF_KC 1")
    print(f"    #define DEF_MR {computation_to_load_optimized[name]['MR']}")
    print(f"    #define DEF_NR {computation_to_load_optimized[name]['NR']}")
    print(f"    #define K_UNROLL {computation_to_load_optimized[name]['K_UNROLL']}")
print("#else")
print("    #error \"Target CPU not supported or defined.\"")
print("#endif")
print("#endif //FMA_LAT")
print("")
print("#ifndef MR")
print("    #define MR DEF_MR")
print("#endif")
print("")
print("#ifndef NR")
print("    #define NR DEF_NR")
print("#endif")
print("")
print("#ifndef KC")
print("    #ifdef DEF_KC")
print("        #define KC DEF_KC")
print("    #endif")
print("#endif")
print("")
print("#ifndef K_UNROLL")
print("    #define K_UNROLL DEF_K_UNROLL")
print("#endif")
print("")
print("_Static_assert(NR % N_VEC == 0, \"NR must be a multiple of N_VEC\");")
print("")
print("#define NV (NR / N_VEC)")
print("")
print("#ifdef USE_DOUBLE")
print("    #error \"Double precision parameters missing. Regenerate using param.py for DP.\"")
print("#endif")
print("")
print("#endif // OUTER_TN_PARAMS_H")
