#!/bin/bash
set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io}"
REPO="${REPO:-$GITHUB_REPOSITORY}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
NAMESPACE="${NAMESPACE:-astroml}"
CANARY_DEPLOYMENT="astroml-api-canary"
STABLE_DEPLOYMENT="astroml-api"
CANARY_SERVICE="astroml-api-canary"
EXPECTED_REPLICAS=1

echo "Deploying canary: ${REGISTRY}/${REPO}:${IMAGE_TAG}"

kubectl apply -f k8s/llm-canary-deployment.yaml -n "${NAMESPACE}"

kubectl set image deployment/${CANARY_DEPLOYMENT} \
  api=${REGISTRY}/${REPO}:${IMAGE_TAG} \
  -n "${NAMESPACE}" --record

kubectl rollout status deployment/${CANARY_DEPLOYMENT} \
  -n "${NAMESPACE}" --timeout=300s

CANARY_REPLICAS=$(kubectl get deployment ${CANARY_DEPLOYMENT} -n "${NAMESPACE}" -o jsonpath='{.status.readyReplicas}' || echo 0)
if [ "${CANARY_REPLICAS}" -ne "${EXPECTED_REPLICAS}" ]; then
  echo "Canary deployment failed: expected ${EXPECTED_REPLICAS} replicas, got ${CANARY_REPLICAS}"
  exit 1
fi

echo "Canary deployed successfully with ${CANARY_REPLICAS} replica(s)"
