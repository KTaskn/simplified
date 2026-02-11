#!/bin/bash
files=(v21_val_fix/train.json
        v21_val_fix/val.json)
for file in "${files[@]}"; do
    echo "Processing $file ..."
    python scripts/tokenize.py "$file" "${file%.json}_unique_tokens.txt" "${file%.json}_unique_tokens_without_standard.txt" --standard_tokens_file TOKENS.txt
done