#!/usr/bin/env sh

OUTPUT="nob"
[ -n "$TAG" ] && OUTPUT="nob-$TAG"

set -xe
gcc -ggdb -o "$OUTPUT" nob.c
