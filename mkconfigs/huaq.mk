TARGET_CPU = TARGET_CPU_KUNPENG920
# XXX: Building with -march=native on Kunpeng-920 expands to wrong extensions on eX3 for gcc15
MARCH=armv8.2-a+dotprod+crc+crypto+fp16fml
