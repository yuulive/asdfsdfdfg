#!/usr/bin/env sh

for file in ./*.md ; do
    if [ -e "$file" ]
    then
        # Remove the starting './'
        f="${f#./}"
        # Remove the ending '.md'
        f="${file%.md}"
        pandoc \
            --standalone \
            --to=html5 \
            --mathml \
            --table-of-contents --toc-depth=2 \
            --template=template.html5 \
            --output=public/"$f".html \
            "$file"
    fi
done
