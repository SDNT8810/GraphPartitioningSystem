#!/usr/bin/env python3
"""
🚀 Proposed_Method Project Runner - Your One-Stop Command Center
==================================================

This script helps you navigate and run the Proposed_Method research system easily.
No more confusion about which file to run!
"""

import os
import sys
import subprocess
from pathlib import Path

class ProjectRunner:
    def __init__(self):
        self.project_root = Path(__file__).parent
        print("🚀 Welcome to Proposed_Method Project Runner!")
        print("=" * 60)
        
    def show_menu(self):
        """Display the main menu with all runnable options."""
        print("\n📋 WHAT DO YOU WANT TO RUN?")
        print("=" * 40)
        print("1️⃣  Main Proposed_Method Demo (RECOMMENDED START HERE)")
        print("2️⃣  Research Validation Framework (Benchmarking)")
        print("3️⃣  IoT Integration Test")
        print("4️⃣  Interactive Dashboard")
        print("5️⃣  Enhanced Features Showcase")
        print("6️⃣  Performance Analysis")
        print("7️⃣  View Generated Reports")
        print("8️⃣  Install Dependencies")
        print("9️⃣  Clean Up Generated Files")
        print("0️⃣  Exit")
        print("=" * 40)
        
    def run_main_demo(self):
        """Run the main Proposed_Method demonstration."""
        print("\n🚀 Running Main Proposed_Method Demo...")
        print("This shows 30 autonomous agents making intelligent decisions!")
        print("-" * 60)
        
        try:
            subprocess.run([sys.executable, "p_md_demo.py"], check=True)
            print("\n✅ Main demo completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running main demo: {e}")
        except FileNotFoundError:
            print("❌ p_md_demo.py not found!")
            
    def run_research_validation(self):
        """Run the research validation framework."""
        print("\n📊 Running Research Validation Framework...")
        print("This generates comprehensive benchmarking and visual reports!")
        print("-" * 60)
        
        try:
            subprocess.run([sys.executable, "research_validation_framework.py"], check=True)
            print("\n✅ Research validation completed!")
            print("📈 Check research_validation_results/ folder for reports")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running research validation: {e}")
        except FileNotFoundError:
            print("❌ research_validation_framework.py not found!")
            
    def run_iot_test(self):
        """Run IoT integration test."""
        print("\n🏭 Running IoT Integration Test...")
        print("This tests real-time industrial IoT data processing!")
        print("-" * 60)
        
        try:
            subprocess.run([sys.executable, "test_iot_integration.py"], check=True)
            print("\n✅ IoT integration test completed!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running IoT test: {e}")
        except FileNotFoundError:
            print("❌ test_iot_integration.py not found!")
            
    def run_dashboard(self):
        """Run interactive dashboard."""
        print("\n📊 Starting Interactive Dashboard...")
        print("This opens a web-based performance dashboard!")
        print("-" * 60)
        print("🌐 Dashboard will open in your browser at http://localhost:8501")
        
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "interactive_dashboard.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running dashboard: {e}")
            print("💡 Try installing streamlit: pip install streamlit")
        except FileNotFoundError:
            print("❌ interactive_dashboard.py not found!")
            
    def run_showcase(self):
        """Run enhanced features showcase."""
        print("\n🎯 Running Enhanced Features Showcase...")
        print("This demonstrates advanced Proposed_Method capabilities!")
        print("-" * 60)
        
        try:
            subprocess.run([sys.executable, "showcase_enhanced_features.py"], check=True)
            print("\n✅ Enhanced features showcase completed!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running showcase: {e}")
        except FileNotFoundError:
            print("❌ showcase_enhanced_features.py not found!")
            
    def run_performance_analysis(self):
        """Run performance analysis."""
        print("\n⚡ Running Performance Analysis...")
        print("This analyzes system performance and optimization!")
        print("-" * 60)
        
        try:
            subprocess.run([sys.executable, "analyze_enhanced_performance.py"], check=True)
            print("\n✅ Performance analysis completed!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running performance analysis: {e}")
        except FileNotFoundError:
            print("❌ analyze_enhanced_performance.py not found!")
            
    def view_reports(self):
        """View generated reports."""
        print("\n📋 Generated Reports and Results:")
        print("-" * 40)
        
        # Check for validation results
        validation_dir = Path("research_validation_results")
        if validation_dir.exists():
            print("✅ Research Validation Results:")
            print(f"   📊 Reports: {validation_dir}/reports/")
            print(f"   📈 Visualizations: {validation_dir}/visualizations/")
            print(f"   💾 Raw Data: {validation_dir}/raw_data/")
            
            # List specific files
            reports_dir = validation_dir / "reports"
            if reports_dir.exists():
                for file in reports_dir.glob("*"):
                    print(f"      - {file.name}")
        else:
            print("❌ No validation results found. Run option 2 first!")
            
        # Check for plots
        plots_dir = Path("plots")
        if plots_dir.exists() and any(plots_dir.iterdir()):
            print("\n✅ Generated Plots:")
            for subdir in plots_dir.iterdir():
                if subdir.is_dir():
                    print(f"   📈 {subdir.name}/")
                    for file in subdir.glob("*.png"):
                        print(f"      - {file.name}")
        else:
            print("❌ No plots found. Run the demos first!")
            
        # Check for logs
        logs_dir = Path("logs")
        if logs_dir.exists() and any(logs_dir.iterdir()):
            print("\n✅ System Logs:")
            for log_file in sorted(logs_dir.glob("*.log"))[-3:]:  # Show last 3 logs
                print(f"   📝 {log_file.name}")
                
    def install_dependencies(self):
        """Install project dependencies."""
        print("\n📦 Installing Dependencies...")
        print("-" * 40)
        
        requirements_file = Path("requirements.txt")
        if requirements_file.exists():
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
                print("\n✅ Dependencies installed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error installing dependencies: {e}")
        else:
            print("❌ requirements.txt not found!")
            print("💡 Installing basic dependencies...")
            basic_deps = ["numpy", "matplotlib", "networkx", "torch", "pandas", "seaborn"]
            for dep in basic_deps:
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
                    print(f"✅ Installed {dep}")
                except:
                    print(f"❌ Failed to install {dep}")
                    
    def cleanup(self):
        """Clean up generated files."""
        print("\n🧹 Cleaning Up Generated Files...")
        print("-" * 40)
        
        cleanup_dirs = ["__pycache__", "logs", "plots", "output", "runs", "research_validation_results"]
        cleanup_files = ["*.log", "*.pyc"]
        
        for dir_name in cleanup_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                import shutil
                try:
                    shutil.rmtree(dir_path)
                    print(f"✅ Removed {dir_name}/")
                except:
                    print(f"❌ Failed to remove {dir_name}/")
                    
        # Remove log files in root
        for log_file in Path(".").glob("*.log"):
            try:
                log_file.unlink()
                print(f"✅ Removed {log_file.name}")
            except:
                print(f"❌ Failed to remove {log_file.name}")
                
        print("\n✅ Cleanup completed!")
        
    def run(self):
        """Main runner loop."""
        while True:
            self.show_menu()
            
            try:
                choice = input("\n🎯 Enter your choice (1-9, or 0 to exit): ").strip()
                
                if choice == "0":
                    print("\n👋 Thanks for using Proposed_Method Project Runner!")
                    break
                elif choice == "1":
                    self.run_main_demo()
                elif choice == "2":
                    self.run_research_validation()
                elif choice == "3":
                    self.run_iot_test()
                elif choice == "4":
                    self.run_dashboard()
                elif choice == "5":
                    self.run_showcase()
                elif choice == "6":
                    self.run_performance_analysis()
                elif choice == "7":
                    self.view_reports()
                elif choice == "8":
                    self.install_dependencies()
                elif choice == "9":
                    self.cleanup()
                else:
                    print("❌ Invalid choice! Please enter 1-9 or 0.")
                    
                input("\n⏸️  Press Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("\n⏸️  Press Enter to continue...")

if __name__ == "__main__":
    runner = ProjectRunner()
    runner.run()
