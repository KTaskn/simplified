#!/bin/bash
# Download MS COCO 2014 images
# train2014: ~13GB, val2014: ~6GB

set -e

DEST_DIR="${1:-./coco_images}"
mkdir -p "$DEST_DIR"

echo "Downloading COCO train2014..."
wget -c http://images.cocodataset.org/zips/train2014.zip -P "$DEST_DIR"

echo "Downloading COCO val2014..."
wget -c http://images.cocodataset.org/zips/val2014.zip -P "$DEST_DIR"

echo "Extracting train2014..."
unzip -q -n "$DEST_DIR/train2014.zip" -d "$DEST_DIR"

echo "Extracting val2014..."
unzip -q -n "$DEST_DIR/val2014.zip" -d "$DEST_DIR"

echo "Done! Images are in $DEST_DIR/train2014/ and $DEST_DIR/val2014/"
