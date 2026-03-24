# Fraud Registry Soroban Contract

A Soroban smart contract that allows "Validators" to post suspected fraudulent account IDs for on-chain reference on the Stellar network.

## 🎯 Purpose

This contract provides a decentralized, on-chain registry for fraud detection where trusted validators can:
- Register as validators with reputation scores
- Report suspicious accounts with evidence and confidence levels
- Achieve consensus on fraudulent accounts through multiple validator reports
- Maintain a transparent, immutable record of fraud reports

## 🏗 Architecture

### Core Components

**Validator Registry**:
- Validator registration with reputation-based access control
- Active/inactive validator management
- Reputation scoring system (0-100)
- Report tracking and accuracy metrics

**Fraud Reporting System**:
- Structured fraud reports with evidence and confidence levels
- Duplicate prevention (one report per validator per account)
- Timestamp and evidence hash storage
- Minimum requirements enforcement (reputation, confidence)

**Consensus Mechanism**:
- Configurable consensus threshold for fraud determination
- Unique validator counting to prevent sybil attacks
- Automatic fraud status determination

### Data Structures

```rust
pub struct FraudReport {
    pub account_id: Address,        // Account being reported
    pub validator: Address,         // Who submitted the report
    pub timestamp: u64,             // When reported
    pub reason: String,            // Reason/evidence
    pub confidence: u8,            // Confidence level (0-100)
    pub evidence_hash: Option<Bytes>, // Evidence data hash
}

pub struct Validator {
    pub address: Address,              // Validator's address
    pub reputation: u8,                // Reputation score (0-100)
    pub report_count: u64,             // Total reports submitted
    pub accurate_reports: u64,         // Verified accurate reports
    pub registration_timestamp: u64,   // When registered
    pub is_active: bool,               // Current status
}
```

## 🚀 Getting Started

### Prerequisites

- Rust 1.70+
- Soroban CLI
- Stellar account with XLM for deployment

### Installation

```bash
# Clone the repository
git clone https://github.com/Traqora/astroml.git
cd astroml

# Install Soroban CLI
cargo install soroban-cli --locked

# Build the contract
cargo build --target wasm32-unknown-unknown --release

# Optimize WASM
soroban contract optimize --wasm target/wasm32-unknown-unknown/release/fraud_registry.wasm
```

### Deployment

```bash
# Deploy to Futurenet (test)
./scripts/deploy.sh <YOUR_SECRET_KEY>

# Deploy to Testnet
./scripts/deploy.sh <YOUR_SECRET_KEY> testnet

# Deploy to Mainnet
./scripts/deploy.sh <YOUR_SECRET_KEY> mainnet
```

## 📋 Contract Functions

### Admin Functions

#### `initialize(admin: Address)`
Initialize the contract with an admin address.

#### `register_validator(admin: Address, validator_address: Address, initial_reputation: u8)`
Register a new validator with an initial reputation score.

#### `update_validator_reputation(admin: Address, validator_address: Address, new_reputation: u8)`
Update a validator's reputation score.

#### `deactivate_validator(admin: Address, validator_address: Address)`
Deactivate a validator (prevent new reports).

#### `update_config(admin: Address, min_reputation: Option<u8>, min_confidence: Option<u8>, consensus_threshold: Option<u8>)`
Update contract configuration parameters.

### Validator Functions

#### `report_fraud(validator: Address, account_id: Address, reason: String, confidence: u8, evidence_hash: Option<Bytes>)`
Submit a fraud report for a suspicious account.

### Query Functions

#### `get_fraud_reports(account_id: Address) -> Vec<FraudReport>`
Get all fraud reports for a specific account.

#### `get_validator(validator_address: Address) -> Result<Validator, Error>`
Get validator information.

#### `is_fraudulent(account_id: Address) -> bool`
Check if an account is considered fraudulent based on consensus.

#### `get_active_validators() -> Vec<Validator>`
Get all currently active validators.

#### `get_config() -> (u8, u8, u8)`
Get current contract configuration (min_reputation, min_confidence, consensus_threshold).

## 🔧 Configuration

The contract is configured with three main parameters:

- **`min_reputation`**: Minimum reputation required to submit reports (default: 50)
- **`min_confidence`**: Minimum confidence level for reports (default: 60)
- **`consensus_threshold`**: Number of unique validators needed for consensus (default: 3)

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_report_fraud

# Run tests with output
cargo test -- --nocapture
```

### Test Coverage

- Contract initialization
- Validator registration and management
- Fraud reporting with validation
- Consensus mechanism
- Error handling
- Configuration updates

## 📖 Usage Examples

### Basic Workflow

```bash
# 1. Deploy contract
./scripts/deploy.sh <ADMIN_SECRET_KEY>

# 2. Register validators
./scripts/examples.sh <CONTRACT_ID> <ADMIN_SECRET_KEY>

# 3. Report fraudulent accounts
soroban contract invoke \
  --id <CONTRACT_ID> \
  --source <VALIDATOR_SECRET_KEY> \
  --network futurenet \
  -- \
  report_fraud \
  --validator <VALIDATOR_ADDRESS> \
  --account_id <SUSPICIOUS_ACCOUNT> \
  --reason "Suspicious transaction patterns" \
  --confidence 85

# 4. Check fraud status
soroban contract invoke \
  --id <CONTRACT_ID> \
  --source <ANY_ACCOUNT> \
  --network futurenet \
  -- \
  is_fraudulent \
  --account_id <SUSPICIOUS_ACCOUNT>
```

### Advanced Configuration

```bash
# Update configuration for stricter requirements
soroban contract invoke \
  --id <CONTRACT_ID> \
  --source <ADMIN_SECRET_KEY> \
  --network futurenet \
  -- \
  update_config \
  --admin <ADMIN_ADDRESS> \
  --min_reputation 70 \
  --min_confidence 80 \
  --consensus_threshold 5
```

## 🔒 Security Considerations

### Validator Management
- Only admin can register/deactivate validators
- Reputation-based access control prevents spam
- Active/inactive status management

### Report Validation
- Minimum reputation requirements
- Confidence level thresholds
- Duplicate prevention per validator
- Evidence hash support for verification

### Consensus Mechanism
- Unique validator counting prevents sybil attacks
- Configurable consensus threshold
- Transparent, immutable reporting

### Access Control
- Admin-only configuration changes
- Validator-only report submission
- Public read access for transparency

## 📊 Integration

### Off-chain Integration
The contract can be integrated with off-chain systems:
- ML fraud detection pipelines
- Exchange monitoring systems
- Compliance platforms
- Risk assessment tools

### On-chain Integration
- Token contracts can check fraud status
- DEX protocols can implement fraud protection
- DeFi protocols can integrate risk scoring

## 🌐 Networks

- **Futurenet**: Testing and development
- **Testnet**: Staging and integration testing
- **Mainnet**: Production deployment

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 🆘 Support

For questions and support:
- Create an issue on GitHub
- Join the Stellar Discord
- Review the Soroban documentation

---

**Built with Soroban SDK for the Stellar Network**
