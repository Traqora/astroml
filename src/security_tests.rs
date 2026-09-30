//! Security-focused tests for the Fraud Registry Soroban contract.
//!
//! This module specifically exercises adversarial scenarios and edge cases
//! identified in SECURITY_AUDIT.md (GitHub Issue #130).  Functional correctness
//! tests live in `src/test.rs`; security tests live here.
//!
//! Run with:
//!   cargo test --lib security  -- --nocapture

#[cfg(test)]
mod security_tests {
    use soroban_sdk::{testutils::Address as _, Address, Bytes, Env, String};
    use crate::{Error, FraudRegistry, FraudRegistryClient};

    // Helper: deploy and initialise a fresh contract instance.
    fn setup_contract(env: &Env) -> (FraudRegistryClient<'_>, Address) {
        let contract_id = env.register_contract(None, FraudRegistry);
        let client = FraudRegistryClient::new(env, &contract_id);
        let admin = Address::generate(env);
        client.initialize(&admin);
        (client, admin)
    }

    // -----------------------------------------------------------------------
    // SC-1 – Re-initialisation attack
    // -----------------------------------------------------------------------

    /// Verify that calling initialize() a second time is prevented.
    /// SC-1 is now remediated with a storage-existence guard.
    #[test]
    fn test_reinitialization_prevented() {
        let env = Env::default();
        let contract_id = env.register_contract(None, FraudRegistry);
        let client = FraudRegistryClient::new(&env, &contract_id);

        let original_admin = Address::generate(&env);
        let attacker = Address::generate(&env);

        client.initialize(&original_admin);

        // Attacker tries to call initialize() again — should now fail with panic.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            client.initialize(&attacker);
        }));
        assert!(result.is_err(), "SC-1: re-initialization should be prevented");

        // Confirm: original admin can still register validators (access preserved).
        let validator = Address::generate(&env);
        let result = client.try_register_validator(&original_admin, &validator, &75_u32);
        assert!(
            result.is_ok(),
            "SC-1: original admin should retain access after re-initialization attempt"
        );
    }

    // -----------------------------------------------------------------------
    // SC-2 – consensus_threshold = 0 marks all accounts as fraudulent
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_consensus_threshold_rejected() {
        let env = Env::default();
        let contract_id = env.register_contract(None, FraudRegistry);
        let client = FraudRegistryClient::new(&env, &contract_id);
        let admin = Address::generate(&env);
        client.initialize(&admin);

        // Set consensus_threshold to 0 — should be rejected.
        // SC-2 is now remediated with a lower-bound check (>= 1) in update_config.
        let result = client.try_update_config(&admin, &None::<u32>, &None::<u32>, &Some(0_u32));
        assert_eq!(
            result,
            Err(Ok(Error::InvalidInput)),
            "SC-2: consensus_threshold = 0 should be rejected"
        );
    }

    // -----------------------------------------------------------------------
    // SC-3 – Boundary values for reputation and confidence
    // -----------------------------------------------------------------------

    #[test]
    fn test_reputation_boundary_101_rejected() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        let validator = Address::generate(&env);

        let result = client.try_register_validator(&admin, &validator, &101_u32);
        assert_eq!(
            result,
            Err(Ok(Error::InvalidInput)),
            "Reputation > 100 must be rejected"
        );
    }

    #[test]
    fn test_reputation_boundary_100_accepted() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        let validator = Address::generate(&env);

        client.register_validator(&admin, &validator, &100_u32);
        let info = client.get_validator(&validator);
        assert_eq!(info.reputation, 100);
    }

    #[test]
    fn test_reputation_boundary_0_accepted() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        let validator = Address::generate(&env);

        client.register_validator(&admin, &validator, &0_u32);
        let info = client.get_validator(&validator);
        assert_eq!(info.reputation, 0);
    }

    #[test]
    fn test_confidence_boundary_100_accepted() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        let validator = Address::generate(&env);
        let target = Address::generate(&env);

        // Use min_confidence = 60 (default); 100 is well within range.
        client.register_validator(&admin, &validator, &75_u32);
        let reason = String::from_str(&env, "High-confidence fraud signal");
        client.report_fraud(&validator, &target, &reason, &100_u32, &None::<Bytes>);

        let reports = client.get_fraud_reports(&target);
        assert_eq!(reports.len(), 1);
        assert_eq!(reports.get_unchecked(0).confidence, 100);
    }

    // -----------------------------------------------------------------------
    // SC-4 – Admin privilege escalation: non-admin cannot update config
    // -----------------------------------------------------------------------

    #[test]
    fn test_non_admin_cannot_update_config() {
        let env = Env::default();
        let (client, _admin) = setup_contract(&env);
        let attacker = Address::generate(&env);

        let result = client.try_update_config(&attacker, &Some(70_u32), &Some(70_u32), &Some(5_u32));
        assert_eq!(
            result,
            Err(Ok(Error::Unauthorized)),
            "Non-admin must not be able to change contract configuration"
        );
    }

    #[test]
    fn test_non_admin_cannot_deactivate_validator() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        let validator = Address::generate(&env);
        let attacker = Address::generate(&env);

        client.register_validator(&admin, &validator, &75_u32);
        let result = client.try_deactivate_validator(&attacker, &validator);
        assert_eq!(
            result,
            Err(Ok(Error::Unauthorized)),
            "Non-admin must not be able to deactivate a validator"
        );
    }

    #[test]
    fn test_non_admin_cannot_update_validator_reputation() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        let validator = Address::generate(&env);
        let attacker = Address::generate(&env);

        client.register_validator(&admin, &validator, &75_u32);
        let result = client.try_update_validator_reputation(&attacker, &validator, &50_u32);
        assert_eq!(
            result,
            Err(Ok(Error::Unauthorized)),
            "Non-admin must not be able to alter validator reputation"
        );
    }

    // -----------------------------------------------------------------------
    // Sybil attack prevention
    // -----------------------------------------------------------------------

    #[test]
    fn test_sybil_single_validator_cannot_manufacture_consensus() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        let validator = Address::generate(&env);
        let target = Address::generate(&env);

        client.register_validator(&admin, &validator, &75_u32);
        let reason = String::from_str(&env, "First report");
        client.report_fraud(&validator, &target, &reason, &80_u32, &None::<Bytes>);

        // Second attempt by same validator must be blocked.
        let reason2 = String::from_str(&env, "Second report attempt");
        let result = client.try_report_fraud(&validator, &target, &reason2, &90_u32, &None::<Bytes>);
        assert_eq!(result, Err(Ok(Error::AlreadyReported)));

        // Account must NOT be flagged — one validator is below the default threshold of 3.
        assert!(!client.is_fraudulent(&target));
    }

    // -----------------------------------------------------------------------
    // Inactive validator cannot escalate via fraud reports
    // -----------------------------------------------------------------------

    #[test]
    fn test_deactivated_validator_cannot_submit_report() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        let validator = Address::generate(&env);
        let target = Address::generate(&env);

        client.register_validator(&admin, &validator, &75_u32);
        client.deactivate_validator(&admin, &validator);

        let reason = String::from_str(&env, "Report from inactive validator");
        let result = client.try_report_fraud(&validator, &target, &reason, &80_u32, &None::<Bytes>);
        assert_eq!(result, Err(Ok(Error::ValidatorNotActive)));
    }

    // -----------------------------------------------------------------------
    // Unregistered validator cannot submit reports
    // -----------------------------------------------------------------------

    #[test]
    fn test_unregistered_address_cannot_submit_report() {
        let env = Env::default();
        let (client, _admin) = setup_contract(&env);
        let ghost = Address::generate(&env);
        let target = Address::generate(&env);

        let reason = String::from_str(&env, "Attempting report without registration");
        let result = client.try_report_fraud(&ghost, &target, &reason, &80_u32, &None::<Bytes>);
        assert_eq!(result, Err(Ok(Error::ValidatorNotFound)));
    }

    // -----------------------------------------------------------------------
    // Confidence below threshold is rejected
    // -----------------------------------------------------------------------

    #[test]
    fn test_low_confidence_report_rejected_at_boundary() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        let validator = Address::generate(&env);
        let target = Address::generate(&env);

        client.register_validator(&admin, &validator, &75_u32);

        // Default min_confidence = 60; value 59 must be rejected.
        let reason = String::from_str(&env, "Borderline evidence");
        let result = client.try_report_fraud(&validator, &target, &reason, &59_u32, &None::<Bytes>);
        assert_eq!(result, Err(Ok(Error::InsufficientConfidence)));

        // Value 60 must be accepted (exactly at threshold).
        client.report_fraud(&validator, &target, &reason, &60_u32, &None::<Bytes>);
    }

    // -----------------------------------------------------------------------
    // Low reputation is rejected at boundary
    // -----------------------------------------------------------------------

    #[test]
    fn test_low_reputation_report_rejected_at_boundary() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        let validator = Address::generate(&env);
        let target = Address::generate(&env);

        // Default min_reputation = 50; register with 49.
        client.register_validator(&admin, &validator, &49_u32);
        let reason = String::from_str(&env, "Low rep attempt");
        let result = client.try_report_fraud(&validator, &target, &reason, &80_u32, &None::<Bytes>);
        assert_eq!(result, Err(Ok(Error::InsufficientReputation)));

        // Now update reputation to exactly 50 — must be accepted.
        client.update_validator_reputation(&admin, &validator, &50_u32);
        client.report_fraud(&validator, &target, &reason, &80_u32, &None::<Bytes>);
    }

    // -----------------------------------------------------------------------
    // Config update: reputation threshold > 100 rejected
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_config_invalid_reputation_threshold() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);

        // min_reputation > 100 is logically impossible and must be rejected.
        let result = client.try_update_config(&admin, &Some(101_u32), &None::<u32>, &None::<u32>);
        assert_eq!(
            result,
            Err(Ok(Error::InvalidInput)),
            "min_reputation > 100 must be rejected in update_config"
        );
    }

    // -----------------------------------------------------------------------
    // SC-3 – Empty reason string validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_reason_string_rejected() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        let validator = Address::generate(&env);
        let target = Address::generate(&env);

        client.register_validator(&admin, &validator, &75_u32);

        // Try to report with empty reason — should be rejected.
        let empty_reason = String::from_str(&env, "");
        let result = client.try_report_fraud(&validator, &target, &empty_reason, &80_u32, &None::<Bytes>);
        assert_eq!(
            result,
            Err(Ok(Error::InvalidInput)),
            "SC-3: empty reason string should be rejected"
        );
    }

    // -----------------------------------------------------------------------
    // Evidence hash: None and Some paths both work
    // -----------------------------------------------------------------------

    #[test]
    fn test_report_with_and_without_evidence_hash() {
        let env = Env::default();
        let (client, admin) = setup_contract(&env);
        let v1 = Address::generate(&env);
        let v2 = Address::generate(&env);
        let target = Address::generate(&env);

        client.register_validator(&admin, &v1, &75_u32);
        client.register_validator(&admin, &v2, &75_u32);

        let reason = String::from_str(&env, "Evidence test");
        let hash = Bytes::from_array(&env, &[0xde, 0xad, 0xbe, 0xef]);

        client.report_fraud(&v1, &target, &reason, &80_u32, &Some(hash));
        client.report_fraud(&v2, &target, &reason, &80_u32, &None::<Bytes>);

        let reports = client.get_fraud_reports(&target);
        assert_eq!(reports.len(), 2);

        let r0 = reports.get_unchecked(0);
        let r1 = reports.get_unchecked(1);

        // One has evidence, one does not — both are stored correctly.
        assert!(
            r0.evidence_hash.is_some() || r1.evidence_hash.is_some(),
            "Expected one report with evidence hash"
        );
        assert!(
            r0.evidence_hash.is_none() || r1.evidence_hash.is_none(),
            "Expected one report without evidence hash"
        );
    }
}
