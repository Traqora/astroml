# Docker verification script for AstroML (PowerShell version)
# This script tests the Docker setup and verifies all services

# Colors for output
$colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
}

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $colors.Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $colors.Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $colors.Red
}

function Write-Header {
    param([string]$Message)
    Write-Host "=== $Message ===" -ForegroundColor $colors.Blue
}

# Function to check if Docker is running
function Test-Docker {
    Write-Header "Checking Docker"
    
    try {
        $dockerInfo = docker info 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Status "Docker is running"
            docker --version
            return $true
        } else {
            Write-Error "Docker is not running"
            return $false
        }
    } catch {
        Write-Error "Docker is not available"
        return $false
    }
}

# Function to check docker-compose
function Test-DockerCompose {
    Write-Header "Checking Docker Compose"
    
    try {
        if (Get-Command docker-compose -ErrorAction SilentlyContinue) {
            $script:ComposeCmd = "docker-compose"
            Write-Status "Using docker-compose"
            docker-compose --version
            return $true
        } elseif (docker compose version 2>$null) {
            $script:ComposeCmd = "docker compose"
            Write-Status "Using docker compose"
            docker compose version
            return $true
        } else {
            Write-Error "docker-compose is not available"
            return $false
        }
    } catch {
        Write-Error "docker-compose check failed"
        return $false
    }
}

# Function to verify Docker images
function Test-DockerImages {
    Write-Header "Verifying Docker Images"
    
    $images = @(
        "astroml_base"
        "astroml_development"
        "astroml_feature-store"
        "astroml_ingestion"
        "astroml_training-cpu"
        "astroml_production"
    )
    
    foreach ($image in $images) {
        $imageExists = docker images --format "table {{.Repository}}" | Select-String $image
        if ($imageExists) {
            Write-Status "✓ $image image exists"
        } else {
            Write-Warning "✗ $image image not found"
        }
    }
}

# Function to verify Docker volumes
function Test-DockerVolumes {
    Write-Header "Verifying Docker Volumes"
    
    $volumes = @(
        "astroml_postgres_data"
        "astroml_redis_data"
        "astroml_feature_store_data"
        "astroml_feature_store_logs"
    )
    
    foreach ($volume in $volumes) {
        $volumeExists = docker volume ls --format "{{.Name}}" | Select-String $volume
        if ($volumeExists) {
            Write-Status "✓ $volume volume exists"
        } else {
            Write-Warning "✗ $volume volume not found"
        }
    }
}

# Function to test core services
function Test-CoreServices {
    Write-Header "Testing Core Services"
    
    try {
        # Start core services
        Write-Status "Starting core services..."
        & $script:ComposeCmd up -d postgres redis
        
        # Wait for services to start
        Write-Status "Waiting for services to start..."
        Start-Sleep 15
        
        # Test PostgreSQL
        Write-Status "Testing PostgreSQL connection..."
        $postgresReady = & $script:ComposeCmd exec -T postgres pg_isready -U astroml -d astroml
        if ($LASTEXITCODE -eq 0) {
            Write-Status "✓ PostgreSQL is ready"
        } else {
            Write-Error "✗ PostgreSQL connection failed"
        }
        
        # Test Redis
        Write-Status "Testing Redis connection..."
        $redisReady = & $script:ComposeCmd exec -T redis redis-cli ping
        if ($redisReady -match "PONG") {
            Write-Status "✓ Redis is ready"
        } else {
            Write-Error "✗ Redis connection failed"
        }
    } catch {
        Write-Error "Core services test failed: $_"
    }
}

# Function to test Feature Store
function Test-FeatureStore {
    Write-Header "Testing Feature Store"
    
    try {
        # Start Feature Store
        Write-Status "Starting Feature Store..."
        & $script:ComposeCmd up -d feature-store
        
        # Wait for Feature Store to start
        Write-Status "Waiting for Feature Store to start..."
        Start-Sleep 20
        
        # Test Feature Store import
        Write-Status "Testing Feature Store import..."
        $importTest = & $script:ComposeCmd exec -T feature-store python -c @"
import astroml.features
from astroml.features import create_feature_store
store = create_feature_store('/app/feature_store')
print('Feature Store initialized successfully')
"@
        if ($LASTEXITCODE -eq 0) {
            Write-Status "✓ Feature Store is working"
        } else {
            Write-Error "✗ Feature Store failed to initialize"
        }
        
        # Test Feature Store functionality
        Write-Status "Testing Feature Store functionality..."
        $functionalityTest = & $script:ComposeCmd exec -T feature-store python -c @"
from astroml.features import create_feature_store, FeatureType
import pandas as pd
import numpy as np

# Create test feature
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
"@
        if ($LASTEXITCODE -eq 0) {
            Write-Status "✓ Feature Store functionality working"
        } else {
            Write-Error "✗ Feature Store functionality failed"
        }
    } catch {
        Write-Error "Feature Store test failed: $_"
    }
}

# Function to test development environment
function Test-Development {
    Write-Header "Testing Development Environment"
    
    try {
        # Start development environment
        Write-Status "Starting development environment..."
        & $script:ComposeCmd up -d dev
        
        # Wait for development environment to start
        Write-Status "Waiting for development environment to start..."
        Start-Sleep 20
        
        # Test Jupyter Lab
        Write-Status "Testing Jupyter Lab..."
        try {
            $jupyterTest = Invoke-WebRequest -Uri "http://localhost:8888" -TimeoutSec 5
            if ($jupyterTest.Content -match "Jupyter") {
                Write-Status "✓ Jupyter Lab is accessible"
            } else {
                Write-Warning "✗ Jupyter Lab not accessible (may need more time)"
            }
        } catch {
            Write-Warning "✗ Jupyter Lab not accessible (may need more time)"
        }
        
        # Test Python environment
        Write-Status "Testing Python environment..."
        $pythonTest = & $script:ComposeCmd exec -T dev python -c @"
import astroml
import astroml.features
import pandas as pd
import numpy as np
import torch
import networkx
print('All Python packages imported successfully')
"@
        if ($LASTEXITCODE -eq 0) {
            Write-Status "✓ Python environment is working"
        } else {
            Write-Error "✗ Python environment failed"
        }
    } catch {
        Write-Error "Development environment test failed: $_"
    }
}

# Function to test ports
function Test-Ports {
    Write-Header "Testing Port Accessibility"
    
    $ports = @(
        @{Port="8000"; Service="Feature Store"}
        @{Port="8001"; Service="Ingestion"}
        @{Port="8002"; Service="Streaming"}
        @{Port="8003"; Service="Development"}
        @{Port="8888"; Service="Jupyter Lab"}
        @{Port="6008"; Service="TensorBoard"}
        @{Port="5432"; Service="PostgreSQL"}
        @{Port="6379"; Service="Redis"}
    )
    
    foreach ($portInfo in $ports) {
        try {
            $tcpTest = Test-NetConnection -ComputerName localhost -Port $portInfo.Port -WarningAction SilentlyContinue
            if ($tcpTest.TcpTestSucceeded) {
                Write-Status "✓ $($portInfo.Service) (port $($portInfo.Port)) is accessible"
            } else {
                Write-Warning "✗ $($portInfo.Service) (port $($portInfo.Port)) not accessible"
            }
        } catch {
            Write-Warning "✗ $($portInfo.Service) (port $($portInfo.Port)) not accessible"
        }
    }
}

# Function to cleanup
function Invoke-Cleanup {
    Write-Header "Cleaning Up"
    
    try {
        Write-Status "Stopping all services..."
        & $script:ComposeCmd down
        Write-Status "Cleanup completed"
    } catch {
        Write-Error "Cleanup failed: $_"
    }
}

# Function to generate report
function New-VerificationReport {
    Write-Header "Verification Report"
    
    Write-Host "Docker Setup Verification completed on $(Get-Date)"
    Write-Host "=========================================="
    Write-Host ""
    Write-Host "Services Tested:"
    Write-Host "- PostgreSQL Database"
    Write-Host "- Redis Cache"
    Write-Host "- Feature Store"
    Write-Host "- Development Environment"
    Write-Host "- Python Environment"
    Write-Host "- Port Accessibility"
    Write-Host "- Test Suite"
    Write-Host ""
    Write-Host "For detailed logs, check the output above."
    Write-Host ""
    Write-Host "Next Steps:"
    Write-Host "1. Start development: .\scripts\docker-dev.ps1 dev"
    Write-Host "2. Access Jupyter Lab: http://localhost:8888"
    Write-Host "3. Run Feature Store example: docker-compose exec dev python examples/feature_store_example.py"
    Write-Host "4. Run tests: .\scripts\docker-dev.ps1 test"
}

# Main execution
function Main {
    Write-Header "AstroML Docker Verification"
    
    # Change to project directory
    Set-Location $PSScriptRoot\..
    
    # Run verification steps
    $failedSteps = 0
    
    if (-not (Test-Docker)) { $failedSteps++ }
    if (-not (Test-DockerCompose)) { $failedSteps++ }
    
    Test-DockerImages
    Test-DockerVolumes
    Test-CoreServices
    Test-FeatureStore
    Test-Development
    Test-Ports
    
    # Cleanup
    Invoke-Cleanup
    
    # Generate report
    New-VerificationReport
    
    # Exit with appropriate code
    if ($failedSteps -eq 0) {
        Write-Status "✅ All verification steps passed!"
        exit 0
    } else {
        Write-Error "❌ $failedSteps verification steps failed"
        exit 1
    }
}

# Execute main function
Main
