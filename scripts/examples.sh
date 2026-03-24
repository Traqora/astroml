#!/bin/bash

# Example usage script for Fraud Registry Contract
# This script demonstrates how to interact with the deployed contract

set -e

# Configuration
CONTRACT_ID="$1"
SECRET_KEY="$2"
NETWORK="futurenet"  # Change to "testnet" or "mainnet" as needed

if [ -z "$CONTRACT_ID" ] || [ -z "$SECRET_KEY" ]; then
    echo "Usage: $0 <CONTRACT_ID> <SECRET_KEY>"
    echo "Please provide the contract ID and your secret key"
    exit 1
fi

echo "🔍 Fraud Registry Contract Examples"
echo "Contract ID: $CONTRACT_ID"
echo "Network: $NETWORK"

# Get current configuration
echo ""
echo "📊 Current Configuration:"
soroban contract invoke \
    --id $CONTRACT_ID \
    --source $SECRET_KEY \
    --network $NETWORK \
    -- \
    get_config

# Register a validator (admin only)
echo ""
echo "👤 Registering Validator..."
VALIDATOR_ADDRESS=$(soroban keys address $SECRET_KEY)
ADMIN_ADDRESS=$VALIDATOR_ADDRESS  # Assuming the deployer is also the admin

soroban contract invoke \
    --id $CONTRACT_ID \
    --source $SECRET_KEY \
    --network $NETWORK \
    -- \
    register_validator \
    --admin $ADMIN_ADDRESS \
    --validator_address $VALIDATOR_ADDRESS \
    --initial_reputation 75

echo "✅ Validator registered: $VALIDATOR_ADDRESS"

# Get validator information
echo ""
echo "📋 Validator Information:"
soroban contract invoke \
    --id $CONTRACT_ID \
    --source $SECRET_KEY \
    --network $NETWORK \
    -- \
    get_validator \
    --validator_address $VALIDATOR_ADDRESS

# Report a fraudulent account
echo ""
echo "⚠️  Reporting Fraudulent Account..."
FRAUDULENT_ACCOUNT="GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB"  # Example address
REASON="Suspicious transaction patterns detected"
CONFIDENCE=85

soroban contract invoke \
    --id $CONTRACT_ID \
    --source $SECRET_KEY \
    --network $NETWORK \
    -- \
    report_fraud \
    --validator $VALIDATOR_ADDRESS \
    --account_id $FRAUDULENT_ACCOUNT \
    --reason "$REASON" \
    --confidence $CONFIDENCE

echo "✅ Fraud report submitted for: $FRAUDULENT_ACCOUNT"

# Check if account is fraudulent
echo ""
echo "🔍 Checking Fraud Status:"
soroban contract invoke \
    --id $CONTRACT_ID \
    --source $SECRET_KEY \
    --network $NETWORK \
    -- \
    is_fraudulent \
    --account_id $FRAUDULENT_ACCOUNT

# Get fraud reports for the account
echo ""
echo "📄 Fraud Reports:"
soroban contract invoke \
    --id $CONTRACT_ID \
    --source $SECRET_KEY \
    --network $NETWORK \
    -- \
    get_fraud_reports \
    --account_id $FRAUDULENT_ACCOUNT

# Get active validators
echo ""
echo "👥 Active Validators:"
soroban contract invoke \
    --id $CONTRACT_ID \
    --source $SECRET_KEY \
    --network $NETWORK \
    -- \
    get_active_validators

echo ""
echo "🎉 Example completed successfully!"
echo ""
echo "Next steps:"
echo "1. Register additional validators to build consensus"
echo "2. Submit more fraud reports to reach consensus threshold"
echo "3. Update configuration as needed"
echo "4. Monitor the contract for new fraud reports"
