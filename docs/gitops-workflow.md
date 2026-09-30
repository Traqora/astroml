# GitOps Model Deployment Workflow

This document describes the automated GitOps workflow for model deployments in AstroML using ArgoCD.

## Overview

The AstroML model deployment pipeline leverages ArgoCD for declarative, Git-centric continuous deployment. All model deployment configurations, including image versions and environment variables, are stored in the `k8s/gitops/` directory.

### Key Components

- **GitOps Manifests**: `k8s/gitops/model-deployment.yaml` defines the model deployment, config map, service, and persistent volume claim.
- **ArgoCD Application**: `k8s/gitops/application.yaml` declares the ArgoCD Application and AppProject to manage syncs and access control.
- **GitHub Actions Workflow**: `.github/workflows/gitops-deploy.yml` orchestrates the CI/CD pipeline, triggering ArgoCD syncs via API and running health checks.
- **GitOps Manager**: `astroml.deployment.gitops_manager.GitOpsManager` provides programmatic access to ArgoCD for sync status, drift detection, deployment approval, and rollback tracking.

## Deployment Procedure

### 1. Initiation
A deployment is initiated manually via the "GitOps Deploy" GitHub Actions workflow or automatically on merges to `main` affecting GitOps configurations.

### 2. CI and Validation
The workflow performs:
- Manifest validation using `kubeconform`.
- Linting and unit tests via `pytest`.
- Building and pushing the model Docker image to the registry.

### 3. Staging Deployment
- The workflow automatically updates the image tag in `k8s/gitops/model-deployment.yaml` and commits it to the repository.
- The `GitOpsManager` API triggers an ArgoCD sync to the staging environment.
- Drift detection ensures the live cluster state matches Git.

### 4. Production Approval
- The workflow pauses and requires manual approval via GitHub Environments.

### 5. Production Deployment
- Upon approval, the workflow triggers the ArgoCD sync to production.
- Health checks are run to verify the deployment's success.

### 6. Rollback
If any step fails after changes are committed to Git, the workflow triggers a rollback via `git revert` to restore the previous state in Git. ArgoCD will then sync the reverted state back to the cluster.

## Monitoring and Drift Remediation

The `GitOpsManager` Python module provides methods to:
- `detect_drift()`: Check if live resources are out of sync with Git.
- `remediate_drift()`: Trigger a self-heal sync to correct drift.
- `get_dashboard()`: Retrieve a summary of recent deployment activity, including phase counts and drift reports.
