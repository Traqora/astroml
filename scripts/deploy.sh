#!/bin/bash

# Deployment script for Fraud Registry Soroban Contract
# This script builds and deploys the contract to the Stellar network

set -e

# Configuration
NETWORK="futurenet"  # Change to "testnet" or "mainnet" as needed
CONTRACT_WASM="target/wasm32-unknown-unknown/release/fraud_registry.wasm"
SECRET_KEY="$1"  # Pass your secret key as first argument

if [ -z "$SECRET_KEY" ]; then
    echo "Usage: $0 <SECRET_KEY>"
    echo "Please provide your Stellar secret key as the first argument"
    exit 1
fi

echo "🚀 Deploying Fraud Registry Contract to $NETWORK"

# Install Rust and Soroban CLI if not present
if ! command -v soroban &> /dev/null; then
    echo "Installing Soroban CLI..."
    cargo install soroban-cli --locked
fi

# Build the contract
echo "📦 Building contract..."
cargo build --target wasm32-unknown-unknown --release

# Optimize the WASM file
echo "⚡ Optimizing WASM..."
soroban contract optimize --wasm $CONTRACT_WASM --output $CONTRACT_WASM

# Deploy the contract
echo "🌐 Deploying contract to $NETWORK..."
CONTRACT_ID=$(soroban contract deploy \
    --wasm $CONTRACT_WASM \
    --source $SECRET_KEY \
    --network $NETWORK)

echo "✅ Contract deployed with ID: $CONTRACT_ID"

# Initialize the contract
echo "🔧 Initializing contract..."
ADMIN_ADDRESS=$(soroban keys address $SECRET_KEY)

soroban contract invoke \
    --id $CONTRACT_ID \
    --source $SECRET_KEY \
    --network $NETWORK \
    -- \
    initialize \
    --admin $ADMIN_ADDRESS

echo "🎉 Contract deployed and initialized successfully!"
echo "Contract ID: $CONTRACT_ID"
echo "Admin Address: $ADMIN_ADDRESS"
echo "Network: $NETWORK"

# Save contract info to a file
cat > contract_info.txt << EOF
Contract ID: $CONTRACT_ID
Admin Address: $ADMIN_ADDRESS
Network: $NETWORK
Deployed at: $(date)
EOF

echo "Contract information saved to contract_info.txt"
