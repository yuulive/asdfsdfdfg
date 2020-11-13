#!/usr/bin/env sh

find ./ -name "*.md" \
    | parallel \
          pandoc \
          --standalone \
          --to=html5 \
          --mathml \
          --table-of-contents --toc-depth=2 \
          --template=template.html5 \
          --output=public/{/.}.html \
          {}

# {} is the argument passed to parallel (file name)
# {/.} is the file name argument passed to parallel, without directory and extension
