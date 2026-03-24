#![no_std]
use soroban_sdk::{contract, contractimpl, contracttype, Address, Env, Map, String, Vec, Symbol, Bytes};

// Contract state storage keys
const DATA_KEY: Symbol = Symbol::short("DATA");

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
    pub confidence: u8,
    /// Evidence data hash (optional)
    pub evidence_hash: Option<Bytes>,
}

#[contracttype]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Validator {
    /// Validator's address
    pub address: Address,
    /// Validator's reputation score (0-100)
    pub reputation: u8,
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
    /// Admin address that can manage validators
    pub admin: Address,
    /// Minimum reputation required to submit reports
    pub min_reputation: u8,
    /// Minimum confidence required for reports
    pub min_confidence: u8,
    /// Number of validators required to mark an account as fraudulent
    pub consensus_threshold: u8,
}

/// Errors that can be returned by the contract
#[contracttype]
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
}

/// Fraud Registry Contract
#[contract]
pub struct FraudRegistry;

#[contractimpl]
impl FraudRegistry {
    /// Initialize the contract with an admin address
    pub fn __init__(env: Env, admin: Address) {
        let data = FraudRegistryData {
            fraud_reports: Map::new(&env),
            validators: Map::new(&env),
            admin: admin.clone(),
            min_reputation: 50, // Default minimum reputation
            min_confidence: 60,  // Default minimum confidence
            consensus_threshold: 3, // Default consensus threshold
        };
        
        env.storage().instance().set(&DATA_KEY, &data);
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
        initial_reputation: u8,
    ) -> Result<(), Error> {
        let mut data = Self::get_data(&env);
        
        // Check if caller is admin
        if data.admin != admin {
            return Err(Error::Unauthorized);
        }
        
        // Check if validator already exists
        if data.validators.contains_key(validator_address) {
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
        confidence: u8,
        evidence_hash: Option<Bytes>,
    ) -> Result<(), Error> {
        let mut data = Self::get_data(&env);
        
        // Check if validator exists and is active
        let validator_info = match data.validators.get(validator) {
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
                if !validators_seen.contains(report.validator) {
                    validators_seen.push_back(report.validator);
                    validator_count += 1;
                }
            }
            
            validator_count >= data.consensus_threshold
        } else {
            false
        }
    }

    /// Get all active validators
    pub fn get_active_validators(env: Env) -> Vec<Validator> {
        let data = Self::get_data(&env);
        let mut active_validators = Vec::new(&env);
        
        for validator in data.validators.values() {
            if validator.is_active {
                active_validators.push_back(validator);
            }
        }
        
        active_validators
    }

    /// Update validator reputation (admin only)
    pub fn update_validator_reputation(
        env: Env,
        admin: Address,
        validator_address: Address,
        new_reputation: u8,
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
        let mut validator = match data.validators.get(validator_address) {
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
        let mut validator = match data.validators.get(validator_address) {
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
        min_reputation: Option<u8>,
        min_confidence: Option<u8>,
        consensus_threshold: Option<u8>,
    ) -> Result<(), Error> {
        let mut data = Self::get_data(&env);
        
        // Check if caller is admin
        if data.admin != admin {
            return Err(Error::Unauthorized);
        }
        
        // Update configuration
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
    pub fn get_config(env: Env) -> (u8, u8, u8) {
        let data = Self::get_data(&env);
        (data.min_reputation, data.min_confidence, data.consensus_threshold)
    }

    /// Helper function to get contract data
    fn get_data(env: &Env) -> FraudRegistryData {
        env.storage().instance().get(&DATA_KEY).unwrap()
    }
}

#[cfg(test)]
mod test;
