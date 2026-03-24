use soroban_sdk::{testutils::Accounts as _, Address, Bytes, Env, String};
use soroban_sdk::contractclient::ContractClient;
use fraud_registry::{Error, FraudRegistry, FraudReport, Validator};

type FraudRegistryClient<'a> = ContractClient<'a, FraudRegistry>;

#[test]
fn test_contract_initialization() {
    let env = Env::default();
    let admin = Address::generate(&env);
    
    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    
    client.initialize(&admin);
    
    // Verify admin is set correctly
    let (min_rep, min_conf, threshold) = client.get_config();
    assert_eq!(min_rep, 50);
    assert_eq!(min_conf, 60);
    assert_eq!(threshold, 3);
}

#[test]
fn test_register_validator() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let validator = Address::generate(&env);
    
    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);
    
    // Register validator
    client.register_validator(&admin, &validator, &75);
    
    // Verify validator registration
    let validator_info = client.get_validator(&validator).unwrap();
    assert_eq!(validator_info.address, validator);
    assert_eq!(validator_info.reputation, 75);
    assert_eq!(validator_info.report_count, 0);
    assert_eq!(validator_info.accurate_reports, 0);
    assert!(validator_info.is_active);
}

#[test]
fn test_register_validator_unauthorized() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let unauthorized = Address::generate(&env);
    let validator = Address::generate(&env);
    
    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);
    
    // Try to register validator with unauthorized account
    let result = client.try_register_validator(&unauthorized, &validator, &75);
    assert_eq!(result, Err(Ok(Error::Unauthorized)));
}

#[test]
fn test_report_fraud() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let validator = Address::generate(&env);
    let fraudulent_account = Address::generate(&env);
    
    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);
    
    // Register validator
    client.register_validator(&admin, &validator, &75);
    
    // Report fraud
    let reason = String::from_str(&env, "Suspicious transaction patterns detected");
    let evidence_hash = Bytes::from_array(&env, &[1, 2, 3, 4, 5]);
    
    client.report_fraud(
        &validator,
        &fraudulent_account,
        &reason,
        &80,
        Some(&evidence_hash),
    );
    
    // Verify fraud report
    let reports = client.get_fraud_reports(&fraudulent_account);
    assert_eq!(reports.len(), 1);
    
    let report = reports.get_unchecked(0);
    assert_eq!(report.account_id, fraudulent_account);
    assert_eq!(report.validator, validator);
    assert_eq!(report.confidence, 80);
    assert_eq!(report.reason, reason);
    assert_eq!(report.evidence_hash, Some(evidence_hash));
}

#[test]
fn test_report_fraud_insufficient_reputation() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let validator = Address::generate(&env);
    let fraudulent_account = Address::generate(&env);
    
    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);
    
    // Register validator with low reputation
    client.register_validator(&admin, &validator, &30);
    
    // Try to report fraud (should fail due to insufficient reputation)
    let reason = String::from_str(&env, "Suspicious transaction patterns");
    let result = client.try_report_fraud(
        &validator,
        &fraudulent_account,
        &reason,
        &80,
        None::<Bytes>,
    );
    assert_eq!(result, Err(Ok(Error::InsufficientReputation)));
}

#[test]
fn test_report_fraud_insufficient_confidence() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let validator = Address::generate(&env);
    let fraudulent_account = Address::generate(&env);
    
    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);
    
    // Register validator
    client.register_validator(&admin, &validator, &75);
    
    // Try to report fraud with low confidence
    let reason = String::from_str(&env, "Suspicious transaction patterns");
    let result = client.try_report_fraud(
        &validator,
        &fraudulent_account,
        &reason,
        &40, // Below minimum confidence of 60
        None::<Bytes>,
    );
    assert_eq!(result, Err(Ok(Error::InsufficientConfidence)));
}

#[test]
fn test_duplicate_report() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let validator = Address::generate(&env);
    let fraudulent_account = Address::generate(&env);
    
    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);
    
    // Register validator
    client.register_validator(&admin, &validator, &75);
    
    // Report fraud first time
    let reason = String::from_str(&env, "Suspicious transaction patterns");
    client.report_fraud(&validator, &fraudulent_account, &reason, &80, None::<Bytes>);
    
    // Try to report fraud again (should fail)
    let result = client.try_report_fraud(&validator, &fraudulent_account, &reason, &80, None::<Bytes>);
    assert_eq!(result, Err(Ok(Error::AlreadyReported)));
}

#[test]
fn test_consensus_threshold() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let validator1 = Address::generate(&env);
    let validator2 = Address::generate(&env);
    let validator3 = Address::generate(&env);
    let fraudulent_account = Address::generate(&env);
    
    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);
    
    // Register validators
    client.register_validator(&admin, &validator1, &75);
    client.register_validator(&admin, &validator2, &75);
    client.register_validator(&admin, &validator3, &75);
    
    // Report fraud with 2 validators (below threshold of 3)
    let reason = String::from_str(&env, "Suspicious transaction patterns");
    client.report_fraud(&validator1, &fraudulent_account, &reason, &80, None::<Bytes>);
    client.report_fraud(&validator2, &fraudulent_account, &reason, &80, None::<Bytes>);
    
    assert!(!client.is_fraudulent(&fraudulent_account));
    
    // Report fraud with 3rd validator (meets threshold)
    client.report_fraud(&validator3, &fraudulent_account, &reason, &80, None::<Bytes>);
    
    assert!(client.is_fraudulent(&fraudulent_account));
}

#[test]
fn test_update_config() {
    let env = Env::default();
    let admin = Address::generate(&env);
    
    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);
    
    // Update configuration
    client.update_config(&admin, Some(&60), Some(&70), Some(&5));
    
    // Verify updated configuration
    let (min_rep, min_conf, threshold) = client.get_config();
    assert_eq!(min_rep, 60);
    assert_eq!(min_conf, 70);
    assert_eq!(threshold, 5);
}

#[test]
fn test_deactivate_validator() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let validator = Address::generate(&env);
    let fraudulent_account = Address::generate(&env);
    
    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);
    
    // Register validator
    client.register_validator(&admin, &validator, &75);
    
    // Deactivate validator
    client.deactivate_validator(&admin, &validator);
    
    // Try to report fraud with deactivated validator (should fail)
    let reason = String::from_str(&env, "Suspicious transaction patterns");
    let result = client.try_report_fraud(&validator, &fraudulent_account, &reason, &80, None::<Bytes>);
    assert_eq!(result, Err(Ok(Error::ValidatorNotActive)));
}

#[test]
fn test_get_active_validators() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let validator1 = Address::generate(&env);
    let validator2 = Address::generate(&env);
    
    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);
    
    // Register validators
    client.register_validator(&admin, &validator1, &75);
    client.register_validator(&admin, &validator2, &75);
    
    // Deactivate one validator
    client.deactivate_validator(&admin, &validator2);
    
    // Get active validators
    let active_validators = client.get_active_validators();
    assert_eq!(active_validators.len(), 1);
    assert_eq!(active_validators.get_unchecked(0).address, validator1);
}
