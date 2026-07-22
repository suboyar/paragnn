import math
def get_micro_tiles(Nvec, Lvfma, Nvfma, regs, name="Unknown"):
    min_product = Lvfma * Nvfma
    tiles = []

    # Iterate through all valid register-constrained values of nv (vector width multiplier)
    # nv starts at 1, and goes up until a micro-tile with mr=1 exceeds the register limit.
    nv = 1
    while True:
        # Minimum mr needed to hide latency for this nv
        min_mr_lat = math.ceil(min_product / nv)

        # The microkernel outer_tn_microkernel_MRxNR_v3 needs by default
        # (mr * nv) + nv + 1 regs. So the max mr allowed by the register file for this nv:
        # regs >= (mr * nv) + nv + 1  ==>  mr <= (regs - nv - 1) / nv
        max_mr_reg = (regs - nv - 1) // nv

        # If even the minimum required mr doesn't fit in registers, then this nv
        # (and any larger nv) is too big.
        if max_mr_reg < min_mr_lat:
            if nv == 1:
                print(f"Warning: Minimum tile for {name} exceeds register limit ({regs} regs)!")
            break

        # Collect all valid mr values for this specific nv
        for mr in range(min_mr_lat, max_mr_reg + 1):
            nr = int(nv * Nvec)
            tiles.append({"mr": mr, "nr": nr, "nv": nv})

        nv += 1
    return tiles

micro_tile_params = {
    # Intel x86-64 (Lvfma and Nvfma gotten from uops.info/table.html)
    "xeonmaxq": {"vlen": 512, "Lvfma": 4, "Nvfma": 2, "regs": 32},   # Sapphire Rapids (Golden Cove), AVX-512
    "habanaq":  {"vlen": 512, "Lvfma": 4, "Nvfma": 2, "regs": 32},   # Ice Lake (Sunny Cove), AVX-512
    "h200q":    {"vlen": 512, "Lvfma": 4, "Nvfma": 2, "regs": 32},   # Granite Rapids (Redwood Cove), AVX-512
    # AMD x86-64 (Lvfma and Nvfma gotten from uops.info/table.html)
    "defq":     {"vlen": 256, "Lvfma": 5, "Nvfma": 2, "regs": 16},   # Naples (Zen), AVX2
    "rome16q":  {"vlen": 256, "Lvfma": 5, "Nvfma": 2, "regs": 16},   # Rome (Zen 2), AVX2
    "fpgaq":    {"vlen": 256, "Lvfma": 4, "Nvfma": 2, "regs": 16},   # Milan A (Zen 3), AVX2
    "milanq":   {"vlen": 256, "Lvfma": 4, "Nvfma": 2, "regs": 16},   # Milan B (Zen 3), AVX2
    "genoaxq":  {"vlen": 512, "Lvfma": 4, "Nvfma": 2, "regs": 32},   # Genoa (Zen 4), AVX-512
    # ARM (Lvfma and Nvfma gotten from uops.info/table.html)
    # https://gitlab.stud.idi.ntnu.no/filipko/sedei/-/blob/main/llvm/lib/Target/AArch64/AArch64SchedThunderX2T99.td#L1176
    "armq":     {"vlen": 128, "Lvfma": 6, "Nvfma": 2, "regs": 32},   # ThunderX2, ASIMD
    # https://gitlab.stud.idi.ntnu.no/filipko/sedei/-/blob/main/llvm/lib/Target/AArch64/AArch64SchedTSV110.td#L665
    "huaq":     {"vlen": 128, "Lvfma": 4, "Nvfma": 2, "regs": 32},   # Kunpeng 920, ASIMD
    # https://gitlab.stud.idi.ntnu.no/filipko/sedei/-/blob/main/llvm/lib/Target/AArch64/AArch64SchedNeoverseV2.td#L1003
    "gh200q":   {"vlen": 128, "Lvfma": 4, "Nvfma": 4, "regs": 32},   # Grace (Neoverse V2), SVE2
}

def select_top_tiles(tiles, regs, top_n=5):
    if not tiles:
        return []

    # Scoring function:
    # 1. Prefer tiles that use close to the max registers (without going over)
    # 2. Prefer larger total tile area (mr * nr) for better reuse
    # 3. Prefer balanced aspect ratios (penalty for extreme imbalance)
    def score_tile(t):
        mr, nr, nv = t['mr'], t['nr'], t['nv']
        used_regs = (mr * nv) + nv + 1
        reg_utilization = used_regs / regs
        area = mr * nr

        # Aspect ratio penalty to avoid extremely lopsided tiles
        aspect_ratio_penalty = abs(mr - nr) / (mr + nr)

        # Combined score (higher is better)
        return (reg_utilization * 10) + (area * 0.1) - (aspect_ratio_penalty * 2)

    # Sort the tiles in descending order based on their score
    sorted_tiles = sorted(tiles, key=score_tile, reverse=True)

    # Return up to the requested top_n tiles
    return sorted_tiles[:top_n]

for name, params in micro_tile_params.items():
    for dtype_bits in [32]:
        Nvec = math.floor(params["vlen"] / dtype_bits)
        tiles = get_micro_tiles(Nvec, params["Lvfma"], params["Nvfma"], params["regs"], name)
        top_tiles = select_top_tiles(tiles, params["regs"])
        # print(f"Top {dtype_bits}-bit tile for {name}: {top_tiles}")
        print(f"{name}:")
        for t in top_tiles:
            mr, nr = t["mr"], t["nr"]
            print(f"  -DMR={mr} -DNR={nr}")
