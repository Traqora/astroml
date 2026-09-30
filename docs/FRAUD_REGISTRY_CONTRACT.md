# Fraud Registry Smart Contract Documentation

## Overview

The Fraud Registry is a Soroban smart contract for the Stellar blockchain that provides a decentralized system for reporting and tracking fraudulent accounts. It uses a validator-based consensus mechanism to ensure reliable fraud detection while maintaining transparency and accountability.

## Table of Contents

1. [Architecture](#architecture)
2. [Data Structures](#data-structures)
3. [Contract Functions](#contract-functions)
4. [Security Features](#security-features)
5. [Usage Examples](#usage-examples)
6. [Deployment Guide](#deployment-guide)
7. [Testing](#testing)
8. [Security Audit](#security-audit)

## Architecture

### Design Principles

- **Validator-Based Consensus**: Multiple validators must agree before an account is marked as fraudulent
- **Reputation System**: Validators have reputation scores that affect their ability to submit reports
- **Appeal Mechanism**: Accounts can appeal fraudulent status with admin review
- **Transparency**: All reports and decisions are publicly visible
- **Security**: Admin-only controls for critical operations

### Key Components

1. **Fraud Reports**: Individual reports submitted by validators
2. **Validators**: Trusted entities with reputation scores
3. **Appeals**: Process for contesting fraudulent status
4. **Consensus Mechanism**: Threshold-based fraud detection

## Data Structures

### FraudReport

Represents a single fraud report submitted by a validator.

```rust
pub struct FraudReport {
    pub account_id: Address,           // Account being reported
    pub validator: Address,            // Validator who submitted report
    pub timestamp: u64,                 // Report timestamp
    pub reason: String,                 // Reason/evidence for fraud
    pub confidence: u32,                // Confidence level (0-100)
    pub evidence_hash: Option<Bytes>,  // Optional evidence hash
}
```

### Validator

Represents a registered validator in the system.

```rust
pub struct Validator {
    pub address: Address,              // Validator's address
    pub reputation: u32,               // Reputation score (0-100)
    pub report_count: u64,             // Total reports submitted
    pub accurate_reports: u64,         // Accurate reports count
    pub registration_timestamp: u64,    // Registration time
    pub is_active: bool,               // Active status
}
```

### Appeal

Represents an appeal against a fraudulent status.

```rust
pub struct Appeal {
    pub account_id: Address,           // Account being appealed
    pub appellant: Address,            // Appellant's address
    pub reason: String,                 // Appeal reason
    pub evidence_hash: Option<Bytes>,  // Evidence hash
    pub timestamp: u64,                 // Appeal timestamp
    pub status: AppealStatus,          // Appeal status
    pub decision_reason: Option<String>, // Admin decision reason
}
```

### AppealStatus

Status of an appeal.

```rust
pub enum AppealStatus {
    Pending = 0,   // Appeal pending review
    Approved = 1,  // Appeal approved (fraud status removed)
    Rejected = 2,  // Appeal rejected (fraud status maintained)
}
```

### FraudRegistryData

Main contract data structure.

```rust
pub struct FraudRegistryData {
    pub fraud_reports: Map<Address, Vec<FraudReport>>,  // Fraud reports
    pub validators: Map<Address, Validator>,             // Validators
    pub appeals: Map<Address, Appeal>,                  // Appeals
    pub admin: Address,                                 // Admin address
    pub min_reputation: u32,                            // Min reputation
    pub min_confidence: u32,                             // Min confidence
    pub consensus_threshold: u32,                       // Consensus threshold
}
```

## Contract Functions

### Initialization

#### `initialize(env: Env, admin: Address) -> Result<(), Error>`

Initializes the contract with an admin address.

**Security Note**: Can only be called once to prevent re-initialization attacks (SC-1).

**Parameters**:
- `env`: Soroban environment
- `admin`: Admin address for contract management

**Errors**:
- `AlreadyInitialized`: Contract already initialized

**Example**:
```rust
let admin = Address::generate(&env);
client.initialize(&admin);
```

### Validator Management

#### `register_validator(env: Env, admin: Address, validator_address: Address, initial_reputation: u32) -> Result<(), Error>`

Registers a new validator (admin only).

**Parameters**:
- `admin`: Admin address
- `validator_address`: Validator address to register
- `initial_reputation`: Initial reputation score (0-100)

**Errors**:
- `Unauthorized`: Caller is not admin
- `ValidatorAlreadyExists`: Validator already registered
- `InvalidInput`: Invalid reputation value

**Example**:
```rust
let validator = Address::generate(&env);
client.register_validator(&admin, &validator, &75);
```

#### `update_validator_reputation(env: Env, admin: Address, validator_address: Address, new_reputation: u32) -> Result<(), Error>`

Updates validator reputation (admin only).

**Parameters**:
- `admin`: Admin address
- `validator_address`: Validator address
- `new_reputation`: New reputation score (0-100)

**Errors**:
- `Unauthorized`: Caller is not admin
- `ValidatorNotFound`: Validator not found
- `InvalidInput`: Invalid reputation value

#### `deactivate_validator(env: Env, admin: Address, validator_address: Address) -> Result<(), Error>`

Deactivates a validator (admin only).

**Parameters**:
- `admin`: Admin address
- `validator_address`: Validator address to deactivate

**Errors**:
- `Unauthorized`: Caller is not admin
- `ValidatorNotFound`: Validator not found

#### `adjust_validator_reputation(env: Env, admin: Address, validator_address: Address, accuracy_delta: i32) -> Result<(), Error>`

Adjusts validator reputation based on report accuracy (admin only).

**Parameters**:
- `admin`: Admin address
- `validator_address`: Validator address
- `accuracy_delta`: Reputation adjustment (-100 to +100)

**Errors**:
- `Unauthorized`: Caller is not admin
- `ValidatorNotFound`: Validator not found
- `InvalidInput`: Invalid delta value

**Example**:
```rust
// Increase reputation for accurate report
client.adjust_validator_reputation(&admin, &validator, &10);

// Decrease reputation for inaccurate report
client.adjust_validator_reputation(&admin, &validator, &-15);
```

#### `batch_register_validators(env: Env, admin: Address, validator_addresses: Vec<Address>, initial_reputations: Vec<u32>) -> Result<(), Error>`

Batch registers multiple validators (admin only).

**Parameters**:
- `admin`: Admin address
- `validator_addresses`: List of validator addresses
- `initial_reputations`: List of initial reputation scores

**Errors**:
- `Unauthorized`: Caller is not admin
- `InvalidInput`: Mismatched array lengths

**Example**:
```rust
let validators = vec![&validator1, &validator2, &validator3];
let reputations = vec![75_u32, 80_u32, 70_u32];
client.batch_register_validators(&admin, validators, reputations);
```

### Fraud Reporting

#### `report_fraud(env: Env, validator: Address, account_id: Address, reason: String, confidence: u32, evidence_hash: Option<Bytes>) -> Result<(), Error>`

Submits a fraud report for an account.

**Parameters**:
- `validator`: Validator address
- `account_id`: Account being reported
- `reason`: Reason/evidence for fraud
- `confidence`: Confidence level (0-100)
- `evidence_hash`: Optional evidence hash

**Errors**:
- `ValidatorNotFound`: Validator not registered
- `ValidatorNotActive`: Validator is inactive
- `InsufficientReputation`: Validator reputation too low
- `InsufficientConfidence`: Confidence below minimum
- `AlreadyReported`: Validator already reported this account

**Example**:
```rust
let reason = String::from_str(&env, "Suspicious transaction patterns");
let evidence = Bytes::from_array(&env, &[1, 2, 3, 4, 5]);
client.report_fraud(&validator, &fraudulent_account, &reason, &85, &Some(evidence));
```

### Query Functions

#### `get_fraud_reports(env: Env, account_id: Address) -> Vec<FraudReport>`

Gets all fraud reports for a specific account.

**Parameters**:
- `account_id`: Account to query

**Returns**: Vector of fraud reports

**Example**:
```rust
let reports = client.get_fraud_reports(&account_id);
```

#### `is_fraudulent(env: Env, account_id: Address) -> bool`

Checks if an account is considered fraudulent based on consensus.

**Parameters**:
- `account_id`: Account to check

**Returns**: Boolean indicating fraudulent status

**Example**:
```rust
let is_fraud = client.is_fraudulent(&account_id);
```

#### `get_validator(env: Env, validator_address: Address) -> Result<Validator, Error>`

Gets validator information.

**Parameters**:
- `validator_address`: Validator address

**Returns**: Validator information

**Errors**:
- `ValidatorNotFound`: Validator not found

#### `get_active_validators(env: Env) -> Vec<Validator>`

Gets all active validators.

**Returns**: Vector of active validators

#### `get_fraudulent_accounts(env: Env) -> Vec<Address>`

Gets all accounts marked as fraudulent.

**Returns**: Vector of fraudulent account addresses

#### `get_statistics(env: Env) -> (u64, u64, u64, u64)`

Gets contract statistics.

**Returns**: Tuple of (total_validators, total_reports, total_fraudulent, total_appeals)

**Example**:
```rust
let (validators, reports, fraudulent, appeals) = client.get_statistics();
```

### Appeal Mechanism

#### `submit_appeal(env: Env, appellant: Address, account_id: Address, reason: String, evidence_hash: Option<Bytes>) -> Result<(), Error>`

Submits an appeal for a fraudulent account.

**Parameters**:
- `appellant`: Appellant address
- `account_id`: Account being appealed
- `reason`: Appeal reason
- `evidence_hash`: Optional evidence hash

**Errors**:
- `InvalidInput`: Account is not fraudulent
- `AppealAlreadyExists`: Appeal already submitted

**Example**:
```rust
let reason = String::from_str(&env, "False positive - legitimate activity");
let evidence = Bytes::from_array(&env, &[6, 7, 8, 9, 10]);
client.submit_appeal(&appellant, &account_id, &reason, &Some(evidence));
```

#### `review_appeal(env: Env, admin: Address, account_id: Address, approve: bool, decision_reason: String) -> Result<(), Error>`

Reviews and decides on an appeal (admin only).

**Parameters**:
- `admin`: Admin address
- `account_id`: Account being appealed
- `approve`: Whether to approve the appeal
- `decision_reason`: Reason for decision

**Errors**:
- `Unauthorized`: Caller is not admin
- `AppealNotFound`: Appeal not found
- `InvalidAppealStatus`: Appeal not pending

**Example**:
```rust
let decision = String::from_str(&env, "Evidence verified - fraud status removed");
client.review_appeal(&admin, &account_id, &true, &decision);
```

#### `get_appeal(env: Env, account_id: Address) -> Result<Appeal, Error>`

Gets appeal information for an account.

**Parameters**:
- `account_id`: Account to query

**Returns**: Appeal information

**Errors**:
- `AppealNotFound`: Appeal not found

### Configuration

#### `update_config(env: Env, admin: Address, min_reputation: Option<u32>, min_confidence: Option<u32>, consensus_threshold: Option<u32>) -> Result<(), Error>`

Updates contract configuration (admin only).

**Parameters**:
- `admin`: Admin address
- `min_reputation`: New minimum reputation (optional)
- `min_confidence`: New minimum confidence (optional)
- `consensus_threshold`: New consensus threshold (optional)

**Errors**:
- `Unauthorized`: Caller is not admin
- `InvalidInput`: Invalid configuration values

**Security Note**: Consensus threshold must be >= 1 to prevent SC-2 vulnerability.

**Example**:
```rust
client.update_config(&admin, &Some(60_u32), &Some(70_u32), &Some(5_u32));
```

#### `get_config(env: Env) -> (u32, u32, u32)`

Gets current contract configuration.

**Returns**: Tuple of (min_reputation, min_confidence, consensus_threshold)

## Security Features

### Implemented Security Measures

1. **Initialization Guard (SC-1 Fixed)**
   - Contract can only be initialized once
   - Prevents re-initialization attacks
   - Returns `AlreadyInitialized` error on subsequent calls

2. **Consensus Threshold Validation (SC-2 Fixed)**
   - Consensus threshold must be >= 1
   - Prevents zero threshold vulnerability
   - Returns `InvalidInput` error for invalid thresholds

3. **Admin Authorization**
   - Critical functions require admin authorization
   - Admin cannot be changed after initialization
   - Prevents unauthorized configuration changes

4. **Validator Reputation System**
   - Validators need minimum reputation to submit reports
   - Reputation can be adjusted based on accuracy
   - Prevents low-quality validators from spamming reports

5. **Sybil Attack Prevention**
   - Each validator can only report an account once
   - Consensus requires multiple independent validators
   - Prevents single validator from manufacturing consensus

6. **Appeal Mechanism**
   - Accounts can appeal fraudulent status
   - Admin review process with documented decisions
   - Provides recourse for false positives

### Security Best Practices

1. **Admin Key Management**
   - Keep admin private key secure
   - Consider multi-sig for critical operations
   - Rotate admin key periodically

2. **Validator Selection**
   - Choose reputable validators
   - Monitor validator performance
   - Remove underperforming validators

3. **Configuration Tuning**
   - Set appropriate consensus threshold
   - Adjust reputation requirements based on network size
   - Monitor false positive/negative rates

## Usage Examples

### Complete Workflow

```rust
use soroban_sdk::{Address, Env, String, Bytes};
use crate::{FraudRegistry, FraudRegistryClient};

// Setup environment
let env = Env::default();
let contract_id = env.register_contract(None, FraudRegistry);
let client = FraudRegistryClient::new(&env, &contract_id);

// Initialize contract
let admin = Address::generate(&env);
client.initialize(&admin);

// Register validators
let validator1 = Address::generate(&env);
let validator2 = Address::generate(&env);
let validator3 = Address::generate(&env);

client.register_validator(&admin, &validator1, &75);
client.register_validator(&admin, &validator2, &80);
client.register_validator(&admin, &validator3, &70);

// Report fraud
let fraudulent_account = Address::generate(&env);
let reason = String::from_str(&env, "Suspicious transaction patterns");
let evidence = Bytes::from_array(&env, &[1, 2, 3, 4, 5]);

client.report_fraud(&validator1, &fraudulent_account, &reason, &85, &Some(evidence));
client.report_fraud(&validator2, &fraudulent_account, &reason, &90, &Some(evidence));
client.report_fraud(&validator3, &fraudulent_account, &reason, &80, &Some(evidence));

// Check if fraudulent
let is_fraudulent = client.is_fraudulent(&fraudulent_account);
assert!(is_fraudulent); // True because 3 validators >= threshold of 3

// Submit appeal
let appellant = Address::generate(&env);
let appeal_reason = String::from_str(&env, "False positive - legitimate business");
let appeal_evidence = Bytes::from_array(&env, &[6, 7, 8, 9, 10]);

client.submit_appeal(&appellant, &fraudulent_account, &appeal_reason, &Some(appeal_evidence));

// Review appeal
let decision = String::from_str(&env, "Evidence verified - removing fraud status");
client.review_appeal(&admin, &fraudulent_account, &true, &decision);

// Verify fraud status removed
let is_fraudulent_after = client.is_fraudulent(&fraudulent_account);
assert!(!is_fraudulent_after);
```

### Batch Validator Registration

```rust
let validators = vec![&validator1, &validator2, &validator3, &validator4];
let reputations = vec![75_u32, 80_u32, 70_u32, 85_u32];

client.batch_register_validators(&admin, validators, reputations);
```

### Reputation Adjustment

```rust
// Reward accurate report
client.adjust_validator_reputation(&admin, &validator1, &10);

// Penalize inaccurate report
client.adjust_validator_reputation(&admin, &validator2, &-20);
```

### Configuration Update

```rust
// Increase consensus threshold for higher security
client.update_config(&admin, &None::<u32>, &None::<u32>, &Some(5_u32));

// Increase minimum reputation requirements
client.update_config(&admin, &Some(70_u32), &None::<u32>, &None::<u32>);
```

## Deployment Guide

### Prerequisites

- Soroban CLI installed
- Rust toolchain installed
- Stellar testnet/mainnet access

### Build Contract

```bash
# Install Soroban CLI
cargo install soroban-cli

# Build contract
soroban contract build

# Optimize contract
soroban contract optimize
```

### Deploy to Testnet

```bash
# Deploy contract
soroban contract deploy \
  --wasm target/wasm/astroml_fraud_registry.wasm \
  --source <admin_secret_key> \
  --network testnet

# Note the contract ID
```

### Initialize Contract

```bash
# Initialize with admin address
soroban contract invoke \
  --id <contract_id> \
  --function initialize \
  --args <admin_address> \
  --source <admin_secret_key> \
  --network testnet
```

### Register First Validator

```bash
# Register validator
soroban contract invoke \
  --id <contract_id> \
  --function register_validator \
  --args <admin_address> <validator_address> <initial_reputation> \
  --source <admin_secret_key> \
  --network testnet
```

### Configuration

```bash
# Update configuration
soroban contract invoke \
  --id <contract_id> \
  --function update_config \
  --args <admin_address> <min_reputation> <min_confidence> <consensus_threshold> \
  --source <admin_secret_key> \
  --network testnet
```

## Testing

### Run All Tests

```bash
# Run functional tests
cargo test --lib

# Run security tests
cargo test --lib security -- --nocapture
```

### Test Coverage

- **Functional Tests**: Core functionality validation
- **Security Tests**: Adversarial scenario testing
- **Boundary Tests**: Edge case validation
- **Integration Tests**: End-to-end workflows

### Security Test Scenarios

1. **SC-1**: Re-initialization attack prevention
2. **SC-2**: Zero consensus threshold validation
3. **SC-3**: Boundary value validation
4. **SC-4**: Admin privilege escalation prevention
5. **Sybil Attack**: Single validator consensus prevention
6. **Inactive Validator**: Deactivated validator prevention
7. **Unregistered Validator**: Unauthorized report prevention

## Security Audit

### Vulnerability Status

| ID | Vulnerability | Status | Fix |
|----|---------------|--------|-----|
| SC-1 | Re-initialization Attack | ✅ Fixed | Initialization guard added |
| SC-2 | Zero Consensus Threshold | ✅ Fixed | Lower bound validation added |

### Security Recommendations

1. **Admin Key Security**
   - Use hardware wallet for admin key
   - Implement multi-sig for critical operations
   - Regular key rotation

2. **Validator Management**
   - Implement validator vetting process
   - Regular performance reviews
   - Clear removal criteria

3. **Monitoring**
   - Monitor report patterns
   - Track validator accuracy
   - Alert on suspicious activity

4. **Governance**
   - Consider DAO for admin functions
   - Implement time-locked admin changes
   - Add emergency pause mechanism

## Error Codes

| Code | Error | Description |
|------|-------|-------------|
| 1 | Unauthorized | Caller lacks required permissions |
| 2 | ValidatorNotFound | Validator not registered |
| 3 | ValidatorNotActive | Validator is inactive |
| 4 | InsufficientReputation | Validator reputation too low |
| 5 | InsufficientConfidence | Report confidence too low |
| 6 | AlreadyReported | Validator already reported this account |
| 7 | InvalidInput | Invalid parameter value |
| 8 | ValidatorAlreadyExists | Validator already registered |
| 9 | AlreadyInitialized | Contract already initialized |
| 10 | AppealNotFound | Appeal not found |
| 11 | AppealAlreadyExists | Appeal already submitted |
| 12 | InvalidAppealStatus | Appeal not in pending state |

## Gas Optimization

### Storage Optimization

- Use efficient data structures (Map, Vec)
- Minimize storage operations
- Batch operations where possible

### Compute Optimization

- Early validation checks
- Efficient iteration patterns
- Avoid unnecessary computations

## Future Enhancements

### Planned Features

1. **Event Logging**
   - Emit events for all state changes
   - Enable off-chain monitoring
   - Improve transparency

2. **Time-Based Expiry**
   - Automatic report expiry
   - Reputation decay over time
   - Appeal time limits

3. **Multi-Sig Admin**
   - Require multiple admin signatures
   - Distributed governance
   - Enhanced security

4. **Staking Mechanism**
   - Validator staking requirements
   - Slashing for malicious behavior
   - Economic incentives

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/Traqora/astroml/issues
- Documentation: https://github.com/Traqora/astroml/docs

## License

This contract is part of the AstroML project and is licensed under the MIT License.
