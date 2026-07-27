#!/bin/bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-astroml}"
STABLE_DEPLOYMENT="${STABLE_DEPLOYMENT:-astroml-api}"
ROLLBACK_TIMEOUT=300

echo "Initiating rollback for ${STABLE_DEPLOYMENT} in namespace ${NAMESPACE}..."

kubectl rollout undo deployment/${STABLE_DEPLOYMENT} -n "${NAMESPACE}"

if kubectl rollout status deployment/${STABLE_DEPLOYMENT} -n "${NAMESPACE}" --timeout=${ROLLBACK_TIMEOUT}s; then
  echo "Rollback completed successfully for ${STABLE_DEPLOYMENT}"
else
  echo "Rollback verification failed for ${STABLE_DEPLOYMENT}"
  kubectl get pods -n "${NAMESPACE}"
  kubectl describe deployment/${STABLE_DEPLOYMENT} -n "${NAMESPACE}"
  exit 1
fi

REVISION=$(kubectl rollout history deployment/${STABLE_DEPLOYMENT} -n "${NAMESPACE}" | tail -n 1 | awk '{print $1}')
echo "Rolled back to revision ${REVISION}"

kubectl get events --field-selector involved-object.name=${STABLE_DEPLOYMENT} -n "${NAMESPACE}" --sort-by='.lastTimestamp' | tail -n 10
