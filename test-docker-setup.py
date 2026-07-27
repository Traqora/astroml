#!/usr/bin/env python3
"""
Docker Setup Verification Script for AstroML
This script tests the Docker setup and verifies all services are working correctly.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def run_command(cmd, timeout=30, capture_output=True):
    """Run a command and return result."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            timeout=timeout,
            capture_output=capture_output,
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def print_header(title):
    """Print a header."""
    print(f"\n{'='*50}")
    print(f"=== {title} ===")
    print('='*50)

def print_success(message):
    """Print success message."""
    print(f"✅ {message}")

def print_error(message):
    """Print error message."""
    print(f"❌ {message}")

def print_warning(message):
    """Print warning message."""
    print(f"⚠️  {message}")

def test_docker():
    """Test if Docker is running."""
    print_header("Testing Docker")
    
    success, stdout, stderr = run_command("docker --version")
    if success:
        print_success(f"Docker is installed: {stdout.strip()}")
        
        success, stdout, stderr = run_command("docker info")
        if success:
            print_success("Docker is running")
            return True
        else:
            print_error("Docker is not running")
            return False
    else:
        print_error("Docker is not installed or not in PATH")
        return False

def test_docker_compose():
    """Test if docker-compose is available."""
    print_header("Testing Docker Compose")
    
    # Try docker-compose first
    success, stdout, stderr = run_command("docker-compose --version")
    if success:
        print_success(f"docker-compose is available: {stdout.strip()}")
        return "docker-compose"
    
    # Try docker compose
    success, stdout, stderr = run_command("docker compose version")
    if success:
        print_success(f"docker compose is available: {stdout.strip()}")
        return "docker compose"
    
    print_error("docker-compose is not available")
    return None

def test_docker_images():
    """Test if Docker images exist."""
    print_header("Testing Docker Images")
    
    images = [
        "astroml_base",
        "astroml_development", 
        "astroml_feature-store",
        "astroml_ingestion",
        "astroml_training-cpu",
        "astroml_production"
    ]
    
    success, stdout, stderr = run_command("docker images")
    if not success:
        print_error("Cannot list Docker images")
        return False
    
    image_list = stdout
    found_images = 0
    
    for image in images:
        if image in image_list:
            print_success(f"{image} image exists")
            found_images += 1
        else:
            print_warning(f"{image} image not found")
    
    print(f"Found {found_images}/{len(images)} images")
    return found_images > 0

def test_core_services():
    """Test core services."""
    print_header("Testing Core Services")
    
    # Start PostgreSQL and Redis
    print("Starting PostgreSQL and Redis...")
    success, stdout, stderr = run_command("docker-compose up -d postgres redis")
    if not success:
        print_error("Failed to start core services")
        return False
    
    # Wait for services to start
    print("Waiting for services to start...")
    time.sleep(15)
    
    # Test PostgreSQL
    print("Testing PostgreSQL connection...")
    success, stdout, stderr = run_command("docker-compose exec -T postgres pg_isready -U astroml -d astroml")
    if success:
        print_success("PostgreSQL is ready")
    else:
        print_error("PostgreSQL connection failed")
    
    # Test Redis
    print("Testing Redis connection...")
    success, stdout, stderr = run_command("docker-compose exec -T redis redis-cli ping")
    if success and "PONG" in stdout:
        print_success("Redis is ready")
    else:
        print_error("Redis connection failed")
    
    return True

def test_feature_store():
    """Test Feature Store."""
    print_header("Testing Feature Store")
    
    # Start Feature Store
    print("Starting Feature Store...")
    success, stdout, stderr = run_command("docker-compose up -d feature-store")
    if not success:
        print_error("Failed to start Feature Store")
        return False
    
    # Wait for Feature Store to start
    print("Waiting for Feature Store to start...")
    time.sleep(20)
    
    # Test Feature Store import
    print("Testing Feature Store import...")
    test_code = """
import astroml.features
from astroml.features import create_feature_store
store = create_feature_store('/app/feature_store')
print('Feature Store initialized successfully')
"""
    
    success, stdout, stderr = run_command(f'docker-compose exec -T feature-store python -c "{test_code}"')
    if success:
        print_success("Feature Store is working")
    else:
        print_error("Feature Store failed to initialize")
        print(f"Error: {stderr}")
    
    # Test Feature Store functionality
    print("Testing Feature Store functionality...")
    functionality_test = """
from astroml.features import create_feature_store, FeatureType
import pandas as pd
import numpy as np

def test_computer(data, entity_col, timestamp_col, **kwargs):
    return pd.DataFrame({'test_feature': [1, 2, 3]})

store = create_feature_store('/app/feature_store')
feature_def = store.register_feature(
    name='test_feature',
    computer=test_computer,
    description='Test feature',
    feature_type=FeatureType.NUMERIC
)
print('Feature registration successful')
"""
    
    success, stdout, stderr = run_command(f'docker-compose exec -T feature-store python -c "{functionality_test}"')
    if success:
        print_success("Feature Store functionality working")
    else:
        print_error("Feature Store functionality failed")
        print(f"Error: {stderr}")
    
    return True

def test_development_environment():
    """Test development environment."""
    print_header("Testing Development Environment")
    
    # Start development environment
    print("Starting development environment...")
    success, stdout, stderr = run_command("docker-compose up -d dev")
    if not success:
        print_error("Failed to start development environment")
        return False
    
    # Wait for development environment to start
    print("Waiting for development environment to start...")
    time.sleep(20)
    
    # Test Python environment
    print("Testing Python environment...")
    python_test = """
import astroml
import astroml.features
import pandas as pd
import numpy as np
try:
    import torch  # noqa: E402
    print('PyTorch imported successfully')
except ImportError:
    print('PyTorch not available')
try:
    import networkx  # noqa: E402
    print('NetworkX imported successfully')
except ImportError:
    print('NetworkX not available')
print('All core Python packages imported successfully')
"""
    
    success, stdout, stderr = run_command(f'docker-compose exec -T dev python -c "{python_test}"')
    if success:
        print_success("Python environment is working")
        print(f"Output: {stdout}")
    else:
        print_error("Python environment failed")
        print(f"Error: {stderr}")
    
    # Test Jupyter Lab accessibility
    print("Testing Jupyter Lab accessibility...")
    try:
        response = requests.get("http://localhost:8888", timeout=5)
        if "Jupyter" in response.text:
            print_success("Jupyter Lab is accessible")
        else:
            print_warning("Jupyter Lab not accessible (may need more time)")
    except requests.exceptions.RequestException:
        print_warning("Jupyter Lab not accessible (may need more time)")
    
    return True

def test_ports():
    """Test port accessibility."""
    print_header("Testing Port Accessibility")
    
    ports = [
        (8000, "Feature Store"),
        (8001, "Ingestion"),
        (8002, "Streaming"),
        (8003, "Development"),
        (8888, "Jupyter Lab"),
        (6008, "TensorBoard"),
        (5432, "PostgreSQL"),
        (6379, "Redis")
    ]
    
    accessible_ports = 0
    
    for port, service in ports:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                print_success(f"{service} (port {port}) is accessible")
                accessible_ports += 1
            else:
                print_warning(f"{service} (port {port}) not accessible")
        except Exception as e:
            print_warning(f"{service} (port {port}) not accessible: {e}")
    
    print(f"Accessible ports: {accessible_ports}/{len(ports)}")
    return accessible_ports > 0

def cleanup():
    """Clean up Docker services."""
    print_header("Cleaning Up")
    
    success, stdout, stderr = run_command("docker-compose down")
    if success:
        print_success("All services stopped")
    else:
        print_error("Failed to stop services")
    
    print("Cleanup completed")

def generate_report():
    """Generate verification report."""
    print_header("Verification Report")
    
    print(f"Docker Setup Verification completed on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    print("")
    print("Services Tested:")
    print("- PostgreSQL Database")
    print("- Redis Cache")
    print("- Feature Store")
    print("- Development Environment")
    print("- Python Environment")
    print("- Port Accessibility")
    print("")
    print("For detailed logs, check the output above.")
    print("")
    print("Next Steps:")
    print("1. Start development: docker-compose --profile dev up -d")
    print("2. Access Jupyter Lab: http://localhost:8888")
    print("3. Run Feature Store example: docker-compose exec dev python examples/feature_store_example.py")
    print("4. Run tests: docker-compose exec dev pytest tests/ -v")

def main():
    """Main verification function."""
    print_header("AstroML Docker Verification")
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    failed_steps = 0
    
    # Run verification steps
    if not test_docker():
        failed_steps += 1
    
    compose_cmd = test_docker_compose()
    if not compose_cmd:
        failed_steps += 1
    
    if not test_docker_images():
        failed_steps += 1
    
    # Only run service tests if Docker is working
    if failed_steps == 0:
        test_core_services()
        test_feature_store()
        test_development_environment()
        test_ports()
    
    # Cleanup
    cleanup()
    
    # Generate report
    generate_report()
    
    # Exit with appropriate code
    if failed_steps == 0:
        print_success("🎉 All verification steps completed!")
        return 0
    else:
        print_error(f"❌ {failed_steps} critical verification steps failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
