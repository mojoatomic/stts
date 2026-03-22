#!/usr/bin/env bash
# Download datasets for STTS validation pipeline.
#
# Usage: bash scripts/download_data.sh [cmapss|battery|pronostia|all]
#
# All datasets are publicly available from NASA and IEEE.
# This script downloads and extracts them to the expected locations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

download_cmapss() {
    echo "=== C-MAPSS Turbofan Engine Degradation ==="
    echo "Source: NASA Prognostics Data Repository"

    local dest="$DATA_DIR/CMAPSSData"
    if [ -f "$dest/train_FD001.txt" ]; then
        echo "Already downloaded: $dest"
        return
    fi

    mkdir -p "$dest"
    local zip="$DATA_DIR/CMAPSSData.zip"

    echo "Downloading..."
    curl -L -o "$zip" \
        "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"

    echo "Extracting..."
    # The zip contains a nested directory; extract and flatten
    local tmpdir="$DATA_DIR/_cmapss_tmp"
    mkdir -p "$tmpdir"
    unzip -o "$zip" -d "$tmpdir"
    # Find the actual data files and move them up
    find "$tmpdir" -name "*.txt" -exec mv {} "$dest/" \;
    find "$tmpdir" -name "*.pdf" -exec mv {} "$dest/" \; 2>/dev/null || true
    rm -rf "$tmpdir" "$zip"

    echo "Done: $(ls "$dest"/*.txt | wc -l | tr -d ' ') files in $dest"
}

download_battery() {
    echo "=== NASA Battery Degradation ==="
    echo "Source: NASA Prognostics Center of Excellence"

    local dest="$DATA_DIR/nasa_battery/extracted"
    if [ -f "$dest/B0005.mat" ]; then
        echo "Already downloaded: $dest"
        return
    fi

    mkdir -p "$dest"
    local zip="$DATA_DIR/nasa_battery/battery_data.zip"

    echo "Downloading..."
    curl -L -o "$zip" \
        "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"

    echo "Extracting outer archive..."
    local tmpdir="$DATA_DIR/nasa_battery/_tmp"
    mkdir -p "$tmpdir"
    unzip -o "$zip" -d "$tmpdir"

    echo "Extracting inner archives..."
    for inner_zip in "$tmpdir"/*/*.zip; do
        unzip -o "$inner_zip" -d "$dest"
    done
    rm -rf "$tmpdir" "$zip"

    echo "Done: $(ls "$dest"/*.mat 2>/dev/null | wc -l | tr -d ' ') .mat files in $dest"
}

download_pronostia() {
    echo "=== PRONOSTIA Bearing Degradation ==="
    echo "Source: IEEE PHM 2012 Prognostics Challenge"
    echo ""
    echo "The PRONOSTIA dataset is available from the IEEE PHM Society:"
    echo "  https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset"
    echo ""
    echo "To download manually:"
    echo "  git clone https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset.git /tmp/pronostia"
    echo "  cp -r /tmp/pronostia/data/* data/pronostia/"
    echo ""

    local dest="$DATA_DIR/pronostia"
    if [ -d "$dest/Learning_set" ]; then
        echo "Already downloaded: $dest"
        return
    fi

    mkdir -p "$dest"

    if command -v git &>/dev/null; then
        echo "Cloning dataset repository..."
        local tmpdir="/tmp/pronostia_download"
        rm -rf "$tmpdir"
        git clone --depth 1 \
            "https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset.git" \
            "$tmpdir"
        cp -r "$tmpdir"/* "$dest/" 2>/dev/null || cp -r "$tmpdir"/data/* "$dest/" 2>/dev/null || true
        rm -rf "$tmpdir"
    else
        echo "git not available. Please download manually (see instructions above)."
        return 1
    fi

    echo "Done: $dest"
}

usage() {
    echo "Usage: $0 [cmapss|battery|pronostia|all]"
    echo ""
    echo "Downloads public datasets for STTS validation:"
    echo "  cmapss    — NASA C-MAPSS turbofan engine degradation (12 MB)"
    echo "  battery   — NASA battery degradation (200 MB)"
    echo "  pronostia — IEEE PHM 2012 bearing degradation (728 MB)"
    echo "  all       — all three datasets"
}

case "${1:-all}" in
    cmapss)    download_cmapss ;;
    battery)   download_battery ;;
    pronostia) download_pronostia ;;
    all)
        download_cmapss
        echo ""
        download_battery
        echo ""
        download_pronostia
        ;;
    -h|--help) usage ;;
    *)
        echo "Unknown dataset: $1"
        usage
        exit 1
        ;;
esac
