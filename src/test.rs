use soroban_sdk::{testutils::Address as _, Address, Bytes, Env, String};
use crate::{Error, FraudRegistry, FraudRegistryClient, AppealStatus};

#[test]
fn test_contract_initialization() {
    let env = Env::default();
    let admin = Address::generate(&env);
    
    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    
    // Initialize should return Ok
    let result = client.try_initialize(&admin);
    assert!(result.is_ok());
    
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
    let validator_info = client.get_validator(&validator);
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
        &Some(evidence_hash.clone()),
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
        &None::<Bytes>,
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
        &None::<Bytes>,
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
    client.report_fraud(&validator, &fraudulent_account, &reason, &80, &None::<Bytes>);
    
    // Try to report fraud again (should fail)
    let result = client.try_report_fraud(&validator, &fraudulent_account, &reason, &80, &None::<Bytes>);
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
    client.report_fraud(&validator1, &fraudulent_account, &reason, &80, &None::<Bytes>);
    client.report_fraud(&validator2, &fraudulent_account, &reason, &80, &None::<Bytes>);
    
    assert!(!client.is_fraudulent(&fraudulent_account));
    
    // Report fraud with 3rd validator (meets threshold)
    client.report_fraud(&validator3, &fraudulent_account, &reason, &80, &None::<Bytes>);
    
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
    client.update_config(&admin, &Some(60_u32), &Some(70_u32), &Some(5_u32));
    
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
    let result = client.try_report_fraud(&validator, &fraudulent_account, &reason, &80, &None::<Bytes>);
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
    let active_validators = client.get_active_validators(&None::<u32>);
    assert_eq!(active_validators.len(), 1);
    assert_eq!(active_validators.get_unchecked(0).address, validator1);
}

#[test]
fn test_initialization_guard() {
    let env = Env::default();
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);

    let admin1 = Address::generate(&env);
    let admin2 = Address::generate(&env);

    // Initialize with first admin
    client.initialize(&admin1);

    // Try to initialize again (should fail with AlreadyInitialized)
    let result = client.try_initialize(&admin2);
    assert_eq!(result, Err(Ok(Error::AlreadyInitialized)));
}

#[test]
fn test_submit_appeal() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let validator1 = Address::generate(&env);
    let validator2 = Address::generate(&env);
    let validator3 = Address::generate(&env);
    let fraudulent_account = Address::generate(&env);
    let appellant = Address::generate(&env);

    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);

    // Register validators
    client.register_validator(&admin, &validator1, &75);
    client.register_validator(&admin, &validator2, &75);
    client.register_validator(&admin, &validator3, &75);

    // Report fraud with 3 validators (meets threshold)
    let reason = String::from_str(&env, "Suspicious transaction patterns");
    client.report_fraud(&validator1, &fraudulent_account, &reason, &80, &None::<Bytes>);
    client.report_fraud(&validator2, &fraudulent_account, &reason, &80, &None::<Bytes>);
    client.report_fraud(&validator3, &fraudulent_account, &reason, &80, &None::<Bytes>);

    // Verify account is fraudulent
    assert!(client.is_fraudulent(&fraudulent_account));

    // Submit appeal
    let appeal_reason = String::from_str(&env, "False positive - legitimate business");
    let evidence_hash = Bytes::from_array(&env, &[1, 2, 3, 4, 5]);
    client.submit_appeal(&appellant, &fraudulent_account, &appeal_reason, &Some(evidence_hash));

    // Verify appeal exists
    let appeal = client.get_appeal(&fraudulent_account);
    assert_eq!(appeal.appellant, appellant);
    assert_eq!(appeal.status, AppealStatus::Pending);
}

#[test]
fn test_submit_appeal_non_fraudulent() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let appellant = Address::generate(&env);
    let non_fraudulent = Address::generate(&env);

    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);

    // Try to appeal non-fraudulent account (should fail)
    let reason = String::from_str(&env, "Appeal reason");
    let result = client.try_submit_appeal(&appellant, &non_fraudulent, &reason, &None::<Bytes>);
    assert_eq!(result, Err(Ok(Error::InvalidInput)));
}

#[test]
fn test_review_appeal_approve() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let validator1 = Address::generate(&env);
    let validator2 = Address::generate(&env);
    let validator3 = Address::generate(&env);
    let fraudulent_account = Address::generate(&env);
    let appellant = Address::generate(&env);

    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);

    // Register validators
    client.register_validator(&admin, &validator1, &75);
    client.register_validator(&admin, &validator2, &75);
    client.register_validator(&admin, &validator3, &75);

    // Report fraud
    let reason = String::from_str(&env, "Suspicious transaction patterns");
    client.report_fraud(&validator1, &fraudulent_account, &reason, &80, &None::<Bytes>);
    client.report_fraud(&validator2, &fraudulent_account, &reason, &80, &None::<Bytes>);
    client.report_fraud(&validator3, &fraudulent_account, &reason, &80, &None::<Bytes>);

    // Submit appeal
    let appeal_reason = String::from_str(&env, "False positive - legitimate business");
    client.submit_appeal(&appellant, &fraudulent_account, &appeal_reason, &None::<Bytes>);

    // Approve appeal
    let decision = String::from_str(&env, "Evidence verified - removing fraud status");
    client.review_appeal(&admin, &fraudulent_account, &true, &decision);

    // Verify fraud status removed
    assert!(!client.is_fraudulent(&fraudulent_account));

    // Verify appeal status updated
    let appeal = client.get_appeal(&fraudulent_account);
    assert_eq!(appeal.status, AppealStatus::Approved);
}

#[test]
fn test_review_appeal_reject() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let validator1 = Address::generate(&env);
    let validator2 = Address::generate(&env);
    let validator3 = Address::generate(&env);
    let fraudulent_account = Address::generate(&env);
    let appellant = Address::generate(&env);

    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);

    // Register validators
    client.register_validator(&admin, &validator1, &75);
    client.register_validator(&admin, &validator2, &75);
    client.register_validator(&admin, &validator3, &75);

    // Report fraud
    let reason = String::from_str(&env, "Suspicious transaction patterns");
    client.report_fraud(&validator1, &fraudulent_account, &reason, &80, &None::<Bytes>);
    client.report_fraud(&validator2, &fraudulent_account, &reason, &80, &None::<Bytes>);
    client.report_fraud(&validator3, &fraudulent_account, &reason, &80, &None::<Bytes>);

    // Submit appeal
    let appeal_reason = String::from_str(&env, "Appeal reason");
    client.submit_appeal(&appellant, &fraudulent_account, &appeal_reason, &None::<Bytes>);

    // Reject appeal
    let decision = String::from_str(&env, "Insufficient evidence");
    client.review_appeal(&admin, &fraudulent_account, &false, &decision);

    // Verify fraud status maintained
    assert!(client.is_fraudulent(&fraudulent_account));

    // Verify appeal status updated
    let appeal = client.get_appeal(&fraudulent_account);
    assert_eq!(appeal.status, AppealStatus::Rejected);
}

#[test]
fn test_adjust_validator_reputation() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let validator = Address::generate(&env);

    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);

    // Register validator
    client.register_validator(&admin, &validator, &75);

    // Increase reputation
    client.adjust_validator_reputation(&admin, &validator, &10);
    let validator_info = client.get_validator(&validator);
    assert_eq!(validator_info.reputation, 85);
    assert_eq!(validator_info.accurate_reports, 1);

    // Decrease reputation
    client.adjust_validator_reputation(&admin, &validator, &-15);
    let validator_info = client.get_validator(&validator);
    assert_eq!(validator_info.reputation, 70);
}

#[test]
fn test_adjust_validator_reputation_bounds() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let validator = Address::generate(&env);

    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);

    // Register validator
    client.register_validator(&admin, &validator, &50);

    // Try to increase beyond 100 (should cap at 100)
    client.adjust_validator_reputation(&admin, &validator, &60);
    let validator_info = client.get_validator(&validator);
    assert_eq!(validator_info.reputation, 100);

    // Try to decrease below 0 (should cap at 0)
    client.adjust_validator_reputation(&admin, &validator, &-150);
    let validator_info = client.get_validator(&validator);
    assert_eq!(validator_info.reputation, 0);
}

#[test]
fn test_batch_register_validators() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let validator1 = Address::generate(&env);
    let validator2 = Address::generate(&env);
    let validator3 = Address::generate(&env);

    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);

    // Batch register validators
    let validators = vec![&validator1, &validator2, &validator3];
    let reputations = vec![75_u32, 80_u32, 70_u32];
    client.batch_register_validators(&admin, validators, reputations);

    // Verify all validators registered
    assert!(client.get_validator(&validator1).is_ok());
    assert!(client.get_validator(&validator2).is_ok());
    assert!(client.get_validator(&validator3).is_ok());
}

#[test]
fn test_get_fraudulent_accounts() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let validator1 = Address::generate(&env);
    let validator2 = Address::generate(&env);
    let validator3 = Address::generate(&env);
    let fraudulent_account = Address::generate(&env);
    let legitimate_account = Address::generate(&env);

    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);

    // Register validators
    client.register_validator(&admin, &validator1, &75);
    client.register_validator(&admin, &validator2, &75);
    client.register_validator(&admin, &validator3, &75);

    // Report fraud on one account
    let reason = String::from_str(&env, "Suspicious transaction patterns");
    client.report_fraud(&validator1, &fraudulent_account, &reason, &80, &None::<Bytes>);
    client.report_fraud(&validator2, &fraudulent_account, &reason, &80, &None::<Bytes>);
    client.report_fraud(&validator3, &fraudulent_account, &reason, &80, &None::<Bytes>);

    // Get fraudulent accounts
    let fraudulent_accounts = client.get_fraudulent_accounts();
    assert_eq!(fraudulent_accounts.len(), 1);
    assert_eq!(fraudulent_accounts.get_unchecked(0), fraudulent_account);
}

#[test]
fn test_get_statistics() {
    let env = Env::default();
    let admin = Address::generate(&env);
    let validator1 = Address::generate(&env);
    let validator2 = Address::generate(&env);
    let fraudulent_account = Address::generate(&env);
    let appellant = Address::generate(&env);

    // Initialize contract
    let contract_id = env.register_contract(None, FraudRegistry);
    let client = FraudRegistryClient::new(&env, &contract_id);
    client.initialize(&admin);

    // Register validators
    client.register_validator(&admin, &validator1, &75);
    client.register_validator(&admin, &validator2, &75);

    // Report fraud
    let reason = String::from_str(&env, "Suspicious transaction patterns");
    client.report_fraud(&validator1, &fraudulent_account, &reason, &80, &None::<Bytes>);
    client.report_fraud(&validator2, &fraudulent_account, &reason, &80, &None::<Bytes>);

    // Submit appeal
    let appeal_reason = String::from_str(&env, "Appeal reason");
    client.submit_appeal(&appellant, &fraudulent_account, &appeal_reason, &None::<Bytes>);

    // Get statistics
    let (validators, reports, fraudulent, appeals) = client.get_statistics();
    assert_eq!(validators, 2);
    assert_eq!(reports, 2);
    assert_eq!(fraudulent, 0); // Below consensus threshold
    assert_eq!(appeals, 1);
}
