#![no_std]
use soroban_sdk::{
    contract, contracterror, contractimpl, contracttype, symbol_short, Address, Bytes, Env, Map,
    String, Symbol, Vec,
};

// Contract state storage keys
const DATA_KEY: Symbol = symbol_short!("DATA");

// Contract type definitions
#[contracttype]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FraudReport {
    /// Account ID being reported as fraudulent
    pub account_id: Address,
    /// Validator who submitted the report
    pub validator: Address,
    /// Timestamp when the report was submitted
    pub timestamp: u64,
    /// Reason/evidence for the fraud report
    pub reason: String,
    /// Confidence level (0-100) of the fraud assessment
    pub confidence: u32,
    /// Evidence data hash (optional)
    pub evidence_hash: Option<Bytes>,
}

#[contracttype]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Validator {
    /// Validator's address
    pub address: Address,
    /// Validator's reputation score (0-100)
    pub reputation: u32,
    /// Number of reports submitted by this validator
    pub report_count: u64,
    /// Number of accurate reports (verified as correct)
    pub accurate_reports: u64,
    /// When the validator was registered
    pub registration_timestamp: u64,
    /// Whether the validator is currently active
    pub is_active: bool,
}

#[contracttype]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FraudRegistryData {
    /// Map of reported accounts to their reports
    pub fraud_reports: Map<Address, Vec<FraudReport>>,
    /// Map of validators to their information
    pub validators: Map<Address, Validator>,
    /// Map of appeals for fraudulent accounts
    pub appeals: Map<Address, Appeal>,
    /// Admin address that can manage validators
    pub admin: Address,
    /// Minimum reputation required to submit reports
    pub min_reputation: u32,
    /// Minimum confidence required for reports
    pub min_confidence: u32,
    /// Number of validators required to mark an account as fraudulent
    pub consensus_threshold: u32,
}

#[contracttype]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Appeal {
    /// Account being appealed
    pub account_id: Address,
    /// Appellant's address
    pub appellant: Address,
    /// Reason for appeal
    pub reason: String,
    /// Evidence hash for appeal
    pub evidence_hash: Option<Bytes>,
    /// Timestamp when appeal was filed
    pub timestamp: u64,
    /// Current status of appeal
    pub status: AppealStatus,
    /// Admin decision reason
    pub decision_reason: Option<String>,
}

#[contracttype]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AppealStatus {
    Pending = 0,
    Approved = 1,
    Rejected = 2,
}

/// Errors that can be returned by the contract
#[contracterror]
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Error {
    /// Unauthorized access
    Unauthorized = 1,
    /// Validator not found
    ValidatorNotFound = 2,
    /// Validator not active
    ValidatorNotActive = 3,
    /// Insufficient reputation
    InsufficientReputation = 4,
    /// Insufficient confidence
    InsufficientConfidence = 5,
    /// Account already reported by this validator
    AlreadyReported = 6,
    /// Invalid input parameters
    InvalidInput = 7,
    /// Validator already exists
    ValidatorAlreadyExists = 8,
    /// Contract already initialized
    AlreadyInitialized = 9,
    /// Appeal not found
    AppealNotFound = 10,
    /// Appeal already exists
    AppealAlreadyExists = 11,
    /// Invalid appeal status
    InvalidAppealStatus = 12,
}

/// Fraud Registry Contract
#[contract]
pub struct FraudRegistry;

#[contractimpl]
impl FraudRegistry {
    /// Initialize the contract with an admin address
    /// 
    /// # Security Note
    /// This function can only be called once. Subsequent calls will fail with
    /// AlreadyInitialized error to prevent re-initialization attacks (SC-1).
    pub fn initialize(env: Env, admin: Address) -> Result<(), Error> {
        // Check if already initialized to prevent re-initialization attack (SC-1)
        if env.storage().instance().has(&DATA_KEY) {
            return Err(Error::AlreadyInitialized);
        }
        let data = FraudRegistryData {
            fraud_reports: Map::new(&env),
            validators: Map::new(&env),
            appeals: Map::new(&env),
            admin: admin.clone(),
            min_reputation: 50, // Default minimum reputation
            min_confidence: 60,  // Default minimum confidence
            consensus_threshold: 3, // Default consensus threshold
        };
        
        env.storage().instance().set(&DATA_KEY, &data);
        Ok(())
    }

    /// Register a new validator
    /// 
    /// # Arguments
    /// * `admin` - The admin address
    /// * `validator_address` - Address of the validator to register
    /// * `initial_reputation` - Initial reputation score (0-100)
    pub fn register_validator(
        env: Env,
        admin: Address,
        validator_address: Address,
        initial_reputation: u32,
    ) -> Result<(), Error> {
        let mut data = Self::get_data(&env);
        
        // Check if caller is admin
        if data.admin != admin {
            return Err(Error::Unauthorized);
        }
        
        // Check if validator already exists
        if data.validators.contains_key(validator_address.clone()) {
            return Err(Error::ValidatorAlreadyExists);
        }

        // Validate reputation
        if initial_reputation > 100 {
            return Err(Error::InvalidInput);
        }

        let validator = Validator {
            address: validator_address.clone(),
            reputation: initial_reputation,
            report_count: 0,
            accurate_reports: 0,
            registration_timestamp: env.ledger().timestamp(),
            is_active: true,
        };
        
        data.validators.set(validator_address, validator);
        env.storage().instance().set(&DATA_KEY, &data);
        
        Ok(())
    }

    /// Submit a fraud report for an account
    /// 
    /// # Arguments
    /// * `validator` - Address of the validator submitting the report
    /// * `account_id` - Address of the account being reported
    /// * `reason` - Reason/evidence for the fraud report
    /// * `confidence` - Confidence level (0-100)
    /// * `evidence_hash` - Optional hash of evidence data
    pub fn report_fraud(
        env: Env,
        validator: Address,
        account_id: Address,
        reason: String,
        confidence: u32,
        evidence_hash: Option<Bytes>,
    ) -> Result<(), Error> {
        let mut data = Self::get_data(&env);
        
        // Validate reason is not empty (SC-3 fix)
        if reason.is_empty() {
            return Err(Error::InvalidInput);
        }
        
        // Check if validator exists and is active
        let validator_info = match data.validators.get(validator.clone()) {
            Some(v) => v,
            None => return Err(Error::ValidatorNotFound),
        };
        
        if !validator_info.is_active {
            return Err(Error::ValidatorNotActive);
        }
        
        // Check reputation and confidence requirements
        if validator_info.reputation < data.min_reputation {
            return Err(Error::InsufficientReputation);
        }
        
        if confidence < data.min_confidence {
            return Err(Error::InsufficientConfidence);
        }
        
        // Check if validator already reported this account
        if let Some(reports) = data.fraud_reports.get(account_id.clone()) {
            for report in reports.iter() {
                if report.validator == validator {
                    return Err(Error::AlreadyReported);
                }
            }
        }
        
        // Create the fraud report
        let report = FraudReport {
            account_id: account_id.clone(),
            validator: validator.clone(),
            timestamp: env.ledger().timestamp(),
            reason: reason.clone(),
            confidence,
            evidence_hash,
        };
        
        // Add the report
        let mut reports = data.fraud_reports.get(account_id.clone()).unwrap_or(Vec::new(&env));
        reports.push_back(report);
        data.fraud_reports.set(account_id, reports);
        
        // Update validator statistics
        let mut updated_validator = validator_info;
        updated_validator.report_count += 1;
        data.validators.set(validator, updated_validator);
        
        env.storage().instance().set(&DATA_KEY, &data);
        
        Ok(())
    }

    /// Get all fraud reports for a specific account
    pub fn get_fraud_reports(env: Env, account_id: Address) -> Vec<FraudReport> {
        let data = Self::get_data(&env);
        data.fraud_reports.get(account_id).unwrap_or(Vec::new(&env))
    }

    /// Get validator information
    pub fn get_validator(env: Env, validator_address: Address) -> Result<Validator, Error> {
        let data = Self::get_data(&env);
        data.validators.get(validator_address).ok_or(Error::ValidatorNotFound)
    }

    /// Check if an account is considered fraudulent based on consensus
    pub fn is_fraudulent(env: Env, account_id: Address) -> bool {
        let data = Self::get_data(&env);
        
        if let Some(reports) = data.fraud_reports.get(account_id) {
            // Count unique validators who reported this account
            let mut validator_count = 0;
            let mut validators_seen = Vec::new(&env);
            
            for report in reports.iter() {
                let report_validator = report.validator.clone();
                if !validators_seen.contains(report_validator.clone()) {
                    validators_seen.push_back(report_validator);
                    validator_count += 1;
                }
            }
            
            validator_count >= data.consensus_threshold
        } else {
            false
        }
    }

    /// Get all active validators (with optional limit to prevent unbounded iteration)
    pub fn get_active_validators(env: Env, limit: Option<u32>) -> Vec<Validator> {
        let data = Self::get_data(&env);
        let mut active_validators = Vec::new(&env);
        let max_count = limit.unwrap_or(100); // Default limit of 100 validators
        let mut count = 0;
        
        for validator in data.validators.values() {
            if validator.is_active {
                if count >= max_count {
                    break;
                }
                active_validators.push_back(validator);
                count += 1;
            }
        }
        
        active_validators
    }

    /// Update validator reputation (admin only)
    pub fn update_validator_reputation(
        env: Env,
        admin: Address,
        validator_address: Address,
        new_reputation: u32,
    ) -> Result<(), Error> {
        let mut data = Self::get_data(&env);
        
        // Check if caller is admin
        if data.admin != admin {
            return Err(Error::Unauthorized);
        }
        
        // Validate reputation
        if new_reputation > 100 {
            return Err(Error::InvalidInput);
        }
        
        // Update validator
        let mut validator = match data.validators.get(validator_address.clone()) {
            Some(v) => v,
            None => return Err(Error::ValidatorNotFound),
        };

        validator.reputation = new_reputation;
        data.validators.set(validator_address, validator);
        
        env.storage().instance().set(&DATA_KEY, &data);
        
        Ok(())
    }

    /// Deactivate a validator (admin only)
    pub fn deactivate_validator(
        env: Env,
        admin: Address,
        validator_address: Address,
    ) -> Result<(), Error> {
        let mut data = Self::get_data(&env);
        
        // Check if caller is admin
        if data.admin != admin {
            return Err(Error::Unauthorized);
        }
        
        // Update validator
        let mut validator = match data.validators.get(validator_address.clone()) {
            Some(v) => v,
            None => return Err(Error::ValidatorNotFound),
        };

        validator.is_active = false;
        data.validators.set(validator_address, validator);
        
        env.storage().instance().set(&DATA_KEY, &data);
        
        Ok(())
    }

    /// Update contract configuration (admin only)
    pub fn update_config(
        env: Env,
        admin: Address,
        min_reputation: Option<u32>,
        min_confidence: Option<u32>,
        consensus_threshold: Option<u32>,
    ) -> Result<(), Error> {
        let mut data = Self::get_data(&env);
        
        // Check if caller is admin
        if data.admin != admin {
            return Err(Error::Unauthorized);
        }
        
        // Validate inputs before applying
        if let Some(rep) = min_reputation {
            if rep > 100 {
                return Err(Error::InvalidInput);
            }
        }
        if let Some(conf) = min_confidence {
            if conf > 100 {
                return Err(Error::InvalidInput);
            }
        }
        if let Some(thresh) = consensus_threshold {
            if thresh == 0 {
                return Err(Error::InvalidInput);
            }
            // Add lower bound check to prevent SC-2 vulnerability
            if thresh < 1 {
                return Err(Error::InvalidInput);
            }
        }

        // Apply configuration
        if let Some(rep) = min_reputation {
            data.min_reputation = rep;
        }
        if let Some(conf) = min_confidence {
            data.min_confidence = conf;
        }
        if let Some(thresh) = consensus_threshold {
            data.consensus_threshold = thresh;
        }
        
        env.storage().instance().set(&DATA_KEY, &data);
        
        Ok(())
    }

    /// Get contract configuration
    pub fn get_config(env: Env) -> (u32, u32, u32) {
        let data = Self::get_data(&env);
        (data.min_reputation, data.min_confidence, data.consensus_threshold)
    }

    /// Submit an appeal for a fraudulent account
    /// 
    /// # Arguments
    /// * `appellant` - Address of the appellant
    /// * `account_id` - Address of the account being appealed
    /// * `reason` - Reason for the appeal
    /// * `evidence_hash` - Optional hash of evidence data
    pub fn submit_appeal(
        env: Env,
        appellant: Address,
        account_id: Address,
        reason: String,
        evidence_hash: Option<Bytes>,
    ) -> Result<(), Error> {
        let mut data = Self::get_data(&env);
        
        // Check if account is marked as fraudulent
        if !Self::is_fraudulent(&env, account_id.clone()) {
            return Err(Error::InvalidInput);
        }
        
        // Check if appeal already exists
        if data.appeals.contains_key(account_id.clone()) {
            return Err(Error::AppealAlreadyExists);
        }
        
        // Create appeal
        let appeal = Appeal {
            account_id: account_id.clone(),
            appellant: appellant.clone(),
            reason: reason.clone(),
            evidence_hash,
            timestamp: env.ledger().timestamp(),
            status: AppealStatus::Pending,
            decision_reason: None,
        };
        
        data.appeals.set(account_id, appeal);
        env.storage().instance().set(&DATA_KEY, &data);
        
        Ok(())
    }

    /// Review and decide on an appeal (admin only)
    /// 
    /// # Arguments
    /// * `admin` - The admin address
    /// * `account_id` - Address of the account being appealed
    /// * `approve` - Whether to approve the appeal
    /// * `decision_reason` - Reason for the decision
    pub fn review_appeal(
        env: Env,
        admin: Address,
        account_id: Address,
        approve: bool,
        decision_reason: String,
    ) -> Result<(), Error> {
        let mut data = Self::get_data(&env);
        
        // Check if caller is admin
        if data.admin != admin {
            return Err(Error::Unauthorized);
        }
        
        // Get appeal
        let mut appeal = match data.appeals.get(account_id.clone()) {
            Some(a) => a,
            None => return Err(Error::AppealNotFound),
        };
        
        // Check if appeal is still pending
        if appeal.status != AppealStatus::Pending {
            return Err(Error::InvalidAppealStatus);
        }
        
        // Update appeal status
        appeal.status = if approve { AppealStatus::Approved } else { AppealStatus::Rejected };
        appeal.decision_reason = Some(decision_reason);
        
        // If approved, remove fraud reports for this account
        if approve {
            data.fraud_reports.remove(account_id.clone());
        }
        
        data.appeals.set(account_id, appeal);
        env.storage().instance().set(&DATA_KEY, &data);
        
        Ok(())
    }

    /// Get appeal information for an account
    pub fn get_appeal(env: Env, account_id: Address) -> Result<Appeal, Error> {
        let data = Self::get_data(&env);
        data.appeals.get(account_id).ok_or(Error::AppealNotFound)
    }

    /// Adjust validator reputation based on report accuracy (admin only)
    /// 
    /// # Arguments
    /// * `admin` - The admin address
    /// * `validator_address` - Address of the validator
    /// * `accuracy_delta` - Reputation adjustment (-100 to +100)
    pub fn adjust_validator_reputation(
        env: Env,
        admin: Address,
        validator_address: Address,
        accuracy_delta: i32,
    ) -> Result<(), Error> {
        let mut data = Self::get_data(&env);
        
        // Check if caller is admin
        if data.admin != admin {
            return Err(Error::Unauthorized);
        }
        
        // Validate delta
        if accuracy_delta < -100 || accuracy_delta > 100 {
            return Err(Error::InvalidInput);
        }
        
        // Get validator
        let mut validator = match data.validators.get(validator_address.clone()) {
            Some(v) => v,
            None => return Err(Error::ValidatorNotFound),
        };
        
        // Adjust reputation with bounds checking
        let new_reputation = if accuracy_delta >= 0 {
            validator.reputation.saturating_add(accuracy_delta as u32)
        } else {
            validator.reputation.saturating_sub((-accuracy_delta) as u32)
        };
        
        validator.reputation = new_reputation.min(100);
        
        // Update accurate reports count if positive adjustment
        if accuracy_delta > 0 {
            validator.accurate_reports += 1;
        }
        
        data.validators.set(validator_address, validator);
        env.storage().instance().set(&DATA_KEY, &data);
        
        Ok(())
    }

    /// Batch register multiple validators (admin only)
    /// 
    /// # Arguments
    /// * `admin` - The admin address
    /// * `validator_addresses` - List of validator addresses
    /// * `initial_reputations` - List of initial reputation scores
    pub fn batch_register_validators(
        env: Env,
        admin: Address,
        validator_addresses: Vec<Address>,
        initial_reputations: Vec<u32>,
    ) -> Result<(), Error> {
        let mut data = Self::get_data(&env);
        
        // Check if caller is admin
        if data.admin != admin {
            return Err(Error::Unauthorized);
        }
        
        // Validate input lengths
        if validator_addresses.len() != initial_reputations.len() {
            return Err(Error::InvalidInput);
        }
        
        // Register each validator
        for i in 0..validator_addresses.len() {
            let validator_address = validator_addresses.get_unchecked(i);
            let initial_reputation = initial_reputations.get_unchecked(i);
            
            // Check if validator already exists
            if data.validators.contains_key(validator_address.clone()) {
                continue; // Skip existing validators
            }
            
            // Validate reputation
            if *initial_reputation > 100 {
                continue; // Skip invalid reputations
            }
            
            let validator = Validator {
                address: validator_address.clone(),
                reputation: *initial_reputation,
                report_count: 0,
                accurate_reports: 0,
                registration_timestamp: env.ledger().timestamp(),
                is_active: true,
            };
            
            data.validators.set(validator_address, validator);
        }
        
        env.storage().instance().set(&DATA_KEY, &data);
        
        Ok(())
    }

    /// Get all fraudulent accounts
    pub fn get_fraudulent_accounts(env: Env) -> Vec<Address> {
        let data = Self::get_data(&env);
        let mut fraudulent_accounts = Vec::new(&env);
        
        for (account_id, _) in data.fraud_reports.iter() {
            if Self::is_fraudulent(&env, account_id.clone()) {
                fraudulent_accounts.push_back(account_id);
            }
        }
        
        fraudulent_accounts
    }

    /// Get contract statistics
    pub fn get_statistics(env: Env) -> (u64, u64, u64, u64) {
        let data = Self::get_data(&env);
        
        let total_validators = data.validators.len() as u64;
        let total_reports = data.fraud_reports.values().fold(0u64, |acc, reports| acc + reports.len());
        let total_fraudulent = Self::get_fraudulent_accounts(env).len() as u64;
        let total_appeals = data.appeals.len() as u64;
        
        (total_validators, total_reports, total_fraudulent, total_appeals)
    }

    /// Helper function to get contract data
    fn get_data(env: &Env) -> FraudRegistryData {
        env.storage().instance().get(&DATA_KEY).unwrap()
    }
}

#[cfg(test)]
mod test;

#[cfg(test)]
mod security_tests;

#[cfg(test)]
mod auth_tests;
