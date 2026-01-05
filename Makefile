# Makefile for Heart Disease MLOps Project
# Cross-platform commands for running experiments and MLflow

.PHONY: help install run-experiments mlflow-ui all clean

help:
	@echo "Heart Disease MLOps - Available Commands"
	@echo "========================================"
	@echo "make install          - Install required Python packages"
	@echo "make run-experiments  - Run ML experiments with MLflow tracking"
	@echo "make mlflow-ui        - Launch MLflow UI"
	@echo "make all              - Run experiments and launch MLflow UI"
	@echo "make clean            - Clean generated files and artifacts"

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Installation complete!"

run-experiments:
	@echo "🚀 Running ML experiments with MLflow..."
	cd notebooks && python feature_engineering_modeling_mlflow.py
	@echo "✅ Experiments complete!"

mlflow-ui:
	@echo "🌐 Launching MLflow UI at http://127.0.0.1:5000"
	@echo "Press Ctrl+C to stop"
	mlflow ui

all: run-experiments
	@echo ""
	@echo "✅ Experiments completed. Press Enter to launch MLflow UI..."
	@read dummy
	@$(MAKE) mlflow-ui

clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf models/*.pkl
	rm -rf results/*.txt results/*.csv results/*.json
	rm -rf screenshots/*.png
	rm -rf mlruns/
	@echo "✅ Clean complete!"
