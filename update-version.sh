#!/usr/bin/env sh

# run as:
#    ./update-version.sh 0.10.0

sed -E -i \
    "s/^version = \"[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+\"/version = \"$1\"/" \
    Cargo.toml

sed -E -i \
    "s/^version: [[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+/version: $1/" \
    design/*.md
