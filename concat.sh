#!/bin/bash

OUTPUT="out.ai"
> "$OUTPUT"

FILES=$(git ls-files | grep -E '\.(py|md|go|sh|toml|yaml|yml|Makefile)$' | grep -v '^out/' | grep -v '^data/' | grep -v '__pycache__')

for f in $FILES; do
  echo "===== $f =====" >> "$OUTPUT"
  cat "$f" >> "$OUTPUT"
  echo -e "\n" >> "$OUTPUT"
done

echo "Written to $OUTPUT"
wc -c "$OUTPUT"
