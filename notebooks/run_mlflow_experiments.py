"""
MLflow Experiment Launcher
Simple Python script to run experiments and launch MLflow UI
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    requirements = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'mlflow',
        'joblib'
    ]

    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {package}")

    print("✅ All packages installed\n")


def run_experiments():
    """Run the ML experiments with MLflow tracking"""
    print("🚀 Running ML experiments with MLflow tracking...")
    print("-" * 60)

    script_path = Path(__file__).parent / "feature_engineering_modeling_mlflow.py"

    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        print("\n✅ Experiments completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running experiments: {e}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Script not found: {script_path}")
        return False


def launch_mlflow_ui():
    """Launch MLflow UI in browser"""
    print("\n" + "=" * 60)
    print("🌐 Launching MLflow UI...")
    print("=" * 60)
    print("\n📊 MLflow UI will open at: http://127.0.0.1:5000")
    print("\n⚠️  Press Ctrl+C to stop the MLflow server\n")

    time.sleep(2)

    try:
        # Open browser
        webbrowser.open('http://127.0.0.1:5000')

        # Start MLflow UI
        subprocess.run(['mlflow', 'ui'], check=True)
    except KeyboardInterrupt:
        print("\n\n✅ MLflow UI stopped")
    except Exception as e:
        print(f"\n❌ Error launching MLflow UI: {e}")
        print("\nYou can manually start it with: mlflow ui")


def main():
    print("=" * 60)
    print("MLflow EXPERIMENT LAUNCHER")
    print("Heart Disease Classification Project")
    print("=" * 60)

    # Menu
    print("\nOptions:")
    print("1. Install requirements")
    print("2. Run experiments (with MLflow tracking)")
    print("3. Launch MLflow UI only")
    print("4. Run experiments + Launch MLflow UI")
    print("5. Exit")

    choice = input("\nEnter your choice (1-5): ").strip()

    if choice == '1':
        install_requirements()
    elif choice == '2':
        run_experiments()
    elif choice == '3':
        launch_mlflow_ui()
    elif choice == '4':
        success = run_experiments()
        if success:
            input("\n✅ Press Enter to launch MLflow UI...")
            launch_mlflow_ui()
    elif choice == '5':
        print("\n👋 Goodbye!")
        sys.exit(0)
    else:
        print("\n❌ Invalid choice. Please run again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        sys.exit(0)
