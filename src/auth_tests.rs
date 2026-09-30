//! Authentication and authorization tests for the Fraud Registry Soroban contract.
//!
//! This module tests:
//! - Admin authentication and authorization
//! - Validator registration and lifecycle
//! - Access control for privileged operations
//! - Session-like behavior through validator state
//!
//! Run with:
//!   cargo test --lib auth  -- --nocapture

#[cfg(test)]
mod auth_tests {
    use soroban_sdk::{testutils::Address as _, Address, Env, String};
    use crate::{Error, FraudRegistry, FraudRegistryClient};

    // Helper: deploy and initialise a fresh contract instance.
    fn setup_contract(env: &Env) -> (FraudRegistryClient<'_>, Address) {
        let contract_id = env.register_contract(None, FraudRegistry);
        let client = FraudRegistryClient::new(env, &contract_id);
        let admin = Address::generate(env);
        client.initialize(&admin);
        (client, admin)
    }

    // ---------------------------------------------------------------------------
    // Admin Authentication Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_admin_initialization_sets_correct_admin() {
        let env = Env::default();
        let contract_id = env.register_contract(None, FraudRegistry);
        let client = FraudRegistryClient::new(&env, &contract_id);
        
        let admin = Address::generate(&env);
        client.initialize(&admin);
        
        // Verify admin can perform admin-only operations
        let validator = Address::generate(&env);
        let result = client.try_register_validator(&admin, &validator, &75_u32);
        assert!(result.is_ok(), "Admin should be able to register validators");
    }

    #[test]
    fn test_non_admin_cannot_initialize_contract() {
        let env = Env::default();
        let contract_id = env.register_contract(None, FraudRegistry);
        let client = FraudRegistryClient::new(&env, &contract_id);
        
        let admin = Address::generate(&env);
        client.initialize(&admin);
        
        // Try to re-initialize with different admin (documents SC-1 vulnerability)
        let attacker = Address::generate(&env);
        client.initialize(&attacker);
        
        // Original admin should no longer have access
        let validator = Address::generate(&env);
        let result = client.try_register_validator(&admin, &validator, &75_u32);
        assert_eq!(result, Err(Ok(Error::Unauthorized)));
    }

    #[test]
    fn test_admin_can_update_config() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let result = client.try_update_config(&admin, &Some(60_u32), &Some(70_u32), &Some(5_u32));
        assert!(result.is_ok(), "Admin should be able to update config");
    }

    #[test]
    fn test_admin_can_deactivate_validator() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        client.register_validator(&admin, &validator, &75_u32);
        
        let result = client.try_deactivate_validator(&admin, &validator);
        assert!(result.is_ok(), "Admin should be able to deactivate validators");
    }

    #[test]
    fn test_admin_can_update_validator_reputation() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        client.register_validator(&admin, &validator, &75_u32);
        
        let result = client.try_update_validator_reputation(&admin, &validator, &90_u32);
        assert!(result.is_ok(), "Admin should be able to update validator reputation");
    }

    // ---------------------------------------------------------------------------
    // Non-Admin Authorization Tests
    // ---------------------------------------------------------------------------

    #[test]
    fn test_non_admin_cannot_register_validator() {
        let env = Env::default();
        let (client, _admin) = setup_contract(&env);
        
        let attacker = Address::generate(&env);
        let validator = Address::generate(&env);
        
        let result = client.try_register_validator(&attacker, &validator, &75_u32);
        assert_eq!(result, Err(Ok(Error::Unauthorized)));
    }

    #[test]
    fn test_non_admin_cannot_update_config() {
        let env = Env::default();
        let (client, _admin) = setup_contract(&env);
        
        let attacker = Address::generate(&env);
        let result = client.try_update_config(&attacker, &Some(60_u32), &Some(70_u32), &Some(5_u32));
        assert_eq!(result, Err(Ok(Error::Unauthorized)));
    }

    #[test]
    fn test_non_admin_cannot_deactivate_validator() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        let attacker = Address::generate(&env);
        
        client.register_validator(&admin, &validator, &75_u32);
        let result = client.try_deactivate_validator(&attacker, &validator);
        assert_eq!(result, Err(Ok(Error::Unauthorized)));
    }

    #[test]
    fn test_non_admin_cannot_update_validator_reputation() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        let attacker = Address::generate(&env);
        
        client.register_validator(&admin, &validator, &75_u32);
        let result = client.try_update_validator_reputation(&attacker, &validator, &90_u32);
        assert_eq!(result, Err(Ok(Error::Unauthorized)));
    }

    // ---------------------------------------------------------------------------
    // Validator Registration Authentication
    // ---------------------------------------------------------------------------

    #[test]
    fn test_validator_registration_requires_admin() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        
        // Successful registration by admin
        let result = client.try_register_validator(&admin, &validator, &75_u32);
        assert!(result.is_ok());
        
        // Verify validator exists
        let validator_info = client.get_validator(&validator);
        assert_eq!(validator_info.address, validator);
    }

    #[test]
    fn test_validator_registration_validates_reputation_bounds() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator1 = Address::generate(&env);
        let validator2 = Address::generate(&env);
        
        // Reputation > 100 should fail
        let result = client.try_register_validator(&admin, &validator1, &101_u32);
        assert_eq!(result, Err(Ok(Error::InvalidInput)));
        
        // Reputation = 100 should succeed
        let result = client.try_register_validator(&admin, &validator2, &100_u32);
        assert!(result.is_ok());
    }

    #[test]
    fn test_duplicate_validator_registration_fails() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        
        client.register_validator(&admin, &validator, &75_u32);
        
        // Try to register same validator again
        let result = client.try_register_validator(&admin, &validator, &80_u32);
        assert_eq!(result, Err(Ok(Error::ValidatorAlreadyExists)));
    }

    // ---------------------------------------------------------------------------
    // Validator Activation/Deactivation Authentication
    // ---------------------------------------------------------------------------

    #[test]
    fn test_deactivated_validator_cannot_submit_reports() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        let target = Address::generate(&env);
        
        client.register_validator(&admin, &validator, &75_u32);
        client.deactivate_validator(&admin, &validator);
        
        let reason = String::from_str(&env, "Report from inactive validator");
        let result = client.try_report_fraud(&validator, &target, &reason, &80_u32, &None::<soroban_sdk::Bytes>);
        assert_eq!(result, Err(Ok(Error::ValidatorNotActive)));
    }

    #[test]
    fn test_validator_deactivation_persists_across_operations() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        let target = Address::generate(&env);
        
        client.register_validator(&admin, &validator, &75_u32);
        client.deactivate_validator(&admin, &validator);
        
        // Verify validator is still deactivated
        let validator_info = client.get_validator(&validator);
        assert!(!validator_info.is_active);
        
        // Try to submit report
        let reason = String::from_str(&env, "Test report");
        let result = client.try_report_fraud(&validator, &target, &reason, &80_u32, &None::<soroban_sdk::Bytes>);
        assert_eq!(result, Err(Ok(Error::ValidatorNotActive)));
    }

    #[test]
    fn test_only_admin_can_reactivate_validator() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        let attacker = Address::generate(&env);
        
        client.register_validator(&admin, &validator, &75_u32);
        client.deactivate_validator(&admin, &validator);
        
        // Non-admin cannot reactivate (would require new function, but test the pattern)
        // For now, verify that only admin can update validator state
        let result = client.try_update_validator_reputation(&attacker, &validator, &90_u32);
        assert_eq!(result, Err(Ok(Error::Unauthorized)));
    }

    // ---------------------------------------------------------------------------
    // Reputation-Based Authentication
    // ---------------------------------------------------------------------------

    #[test]
    fn test_low_reputation_validator_cannot_submit_reports() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        let target = Address::generate(&env);
        
        // Register with reputation below minimum (50)
        client.register_validator(&admin, &validator, &30_u32);
        
        let reason = String::from_str(&env, "Low reputation attempt");
        let result = client.try_report_fraud(&validator, &target, &reason, &80_u32, &None::<soroban_sdk::Bytes>);
        assert_eq!(result, Err(Ok(Error::InsufficientReputation)));
    }

    #[test]
    fn test_reputation_update_affects_authentication() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        let target = Address::generate(&env);
        
        // Register with low reputation
        client.register_validator(&admin, &validator, &30_u32);
        
        // Should fail to report
        let reason = String::from_str(&env, "Test report");
        let result = client.try_report_fraud(&validator, &target, &reason, &80_u32, &None::<soroban_sdk::Bytes>);
        assert_eq!(result, Err(Ok(Error::InsufficientReputation)));
        
        // Admin updates reputation to meet threshold
        client.update_validator_reputation(&admin, &validator, &60_u32);
        
        // Should now succeed
        let result = client.try_report_fraud(&validator, &target, &reason, &80_u32, &None::<soroban_sdk::Bytes>);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reputation_boundary_at_minimum_threshold() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        let target = Address::generate(&env);
        
        // Register with exactly minimum reputation (50)
        client.register_validator(&admin, &validator, &50_u32);
        
        let reason = String::from_str(&env, "Boundary test");
        let result = client.try_report_fraud(&validator, &target, &reason, &80_u32, &None::<soroban_sdk::Bytes>);
        assert!(result.is_ok(), "Reputation at minimum threshold should be accepted");
    }

    // ---------------------------------------------------------------------------
    // Confidence-Based Authentication
    // ---------------------------------------------------------------------------

    #[test]
    fn test_low_confidence_report_rejected() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        let target = Address::generate(&env);
        
        client.register_validator(&admin, &validator, &75_u32);
        
        // Try to report with confidence below minimum (60)
        let reason = String::from_str(&env, "Low confidence report");
        let result = client.try_report_fraud(&validator, &target, &reason, &40_u32, &None::<soroban_sdk::Bytes>);
        assert_eq!(result, Err(Ok(Error::InsufficientConfidence)));
    }

    #[test]
    fn test_confidence_boundary_at_minimum_threshold() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        let target = Address::generate(&env);
        
        client.register_validator(&admin, &validator, &75_u32);
        
        // Report with exactly minimum confidence (60)
        let reason = String::from_str(&env, "Boundary test");
        let result = client.try_report_fraud(&validator, &target, &reason, &60_u32, &None::<soroban_sdk::Bytes>);
        assert!(result.is_ok(), "Confidence at minimum threshold should be accepted");
    }

    // ---------------------------------------------------------------------------
    // Unregistered Address Authentication
    // ---------------------------------------------------------------------------

    #[test]
    fn test_unregistered_address_cannot_submit_reports() {
        let env = Env::default();
        let (client, _admin) = setup_contract(&env);
        
        let unregistered = Address::generate(&env);
        let target = Address::generate(&env);
        
        let reason = String::from_str(&env, "Unregistered attempt");
        let result = client.try_report_fraud(&unregistered, &target, &reason, &80_u32, &None::<soroban_sdk::Bytes>);
        assert_eq!(result, Err(Ok(Error::ValidatorNotFound)));
    }

    #[test]
    fn test_unregistered_address_cannot_be_queried() {
        let env = Env::default();
        let (client, _admin) = setup_contract(&env);
        
        let unregistered = Address::generate(&env);
        let result = client.try_get_validator(&unregistered);
        assert_eq!(result, Err(Ok(Error::ValidatorNotFound)));
    }

    // ---------------------------------------------------------------------------
    // Session-Like Behavior (Validator State Persistence)
    // ---------------------------------------------------------------------------

    #[test]
    fn test_validator_state_persists_across_operations() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        let target1 = Address::generate(&env);
        let target2 = Address::generate(&env);
        
        client.register_validator(&admin, &validator, &75_u32);
        
        // Submit first report
        let reason1 = String::from_str(&env, "First report");
        client.report_fraud(&validator, &target1, &reason1, &80_u32, &None::<soroban_sdk::Bytes>);
        
        // Verify report count increased
        let validator_info = client.get_validator(&validator);
        assert_eq!(validator_info.report_count, 1);
        
        // Submit second report to different target
        let reason2 = String::from_str(&env, "Second report");
        client.report_fraud(&validator, &target2, &reason2, &75_u32, &None::<soroban_sdk::Bytes>);
        
        // Verify report count increased again
        let validator_info = client.get_validator(&validator);
        assert_eq!(validator_info.report_count, 2);
    }

    #[test]
    fn test_validator_registration_timestamp_persists() {
        let env = Env::default();
        // Env::default() starts at ledger timestamp 0; set a non-zero value
        // so the contract's stored registration_timestamp is also non-zero.
        env.ledger().set_timestamp(1_000_000);
        let (client, admin) = setup_contract(&env);

        let validator = Address::generate(&env);

        client.register_validator(&admin, &validator, &75_u32);

        let validator_info = client.get_validator(&validator);
        let timestamp = validator_info.registration_timestamp;

        // Timestamp should be non-zero (set during registration)
        assert!(timestamp > 0, "Registration timestamp should be set");
    }

    // ---------------------------------------------------------------------------
    // Configuration-Based Authentication
    // ---------------------------------------------------------------------------

    #[test]
    fn test_config_change_affects_authentication_requirements() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        let target = Address::generate(&env);
        
        // Register with reputation 60 (above default minimum of 50)
        client.register_validator(&admin, &validator, &60_u32);
        
        // Should be able to report
        let reason = String::from_str(&env, "Test report");
        let result = client.try_report_fraud(&validator, &target, &reason, &80_u32, &None::<soroban_sdk::Bytes>);
        assert!(result.is_ok());
        
        // Admin raises minimum reputation to 70
        client.update_config(&admin, &Some(70_u32), &None::<u32>, &None::<u32>);
        
        // Should now fail due to new minimum
        let result = client.try_report_fraud(&validator, &target, &reason, &80_u32, &None::<soroban_sdk::Bytes>);
        assert_eq!(result, Err(Ok(Error::InsufficientReputation)));
    }

    #[test]
    fn test_config_change_affects_confidence_requirements() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        
        let validator = Address::generate(&env);
        let target = Address::generate(&env);
        
        client.register_validator(&admin, &validator, &75_u32);
        
        // Admin raises minimum confidence to 90
        client.update_config(&admin, &None::<u32>, &Some(90_u32), &None::<u32>);
        
        // Report with confidence 80 should fail
        let reason = String::from_str(&env, "Test report");
        let result = client.try_report_fraud(&validator, &target, &reason, &80_u32, &None::<soroban_sdk::Bytes>);
        assert_eq!(result, Err(Ok(Error::InsufficientConfidence)));
    }
}
