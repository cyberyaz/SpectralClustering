#!/usr/bin/env bash

echo "Generating data..."
python3 src/step1_data.py
echo "Creating graph..."
python3 src/step2_graph.py
echo "Spectral clustering..."
python3 src/step3_4_spectral.py
echo "Other methods and comparison..."
python3 src/baseline_and_metrics.py
echo "Plotting..."
python3 src/quick_plot.py
