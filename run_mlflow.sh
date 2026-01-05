#!/bin/bash

# Quick Start Script for MLflow Experiment Tracking
# Heart Disease MLOps Project - Task 3

echo "=========================================="
echo "MLflow Experiment Tracking - Quick Start"
echo "Heart Disease Classification"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -d "notebooks" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "Choose an option:"
echo "  1) Run main feature engineering + MLflow (Logistic Regression & Random Forest)"
echo "  2) Run comprehensive MLflow experiments (8 different models)"
echo "  3) Start MLflow UI only"
echo "  4) Run all experiments then start MLflow UI"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Running feature engineering with MLflow tracking..."
        cd notebooks
        python feature_engineering_modeling.py
        echo ""
        echo "✅ Complete! To view results, run: mlflow ui"
        ;;
    2)
        echo ""
        echo "🚀 Running comprehensive MLflow experiments..."
        cd notebooks
        python mlflow_experiments.py
        echo ""
        echo "✅ Complete! To view results, run: mlflow ui"
        ;;
    3)
        echo ""
        echo "🌐 Starting MLflow UI..."
        echo "   Access at: http://localhost:5000"
        echo "   Press Ctrl+C to stop"
        echo ""
        cd notebooks
        mlflow ui
        ;;
    4)
        echo ""
        echo "🚀 Running comprehensive MLflow experiments..."
        cd notebooks
        python mlflow_experiments.py
        echo ""
        echo "✅ Experiments complete!"
        echo ""
        echo "🌐 Starting MLflow UI..."
        echo "   Access at: http://localhost:5000"
        echo "   Press Ctrl+C to stop"
        echo ""
        mlflow ui
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
