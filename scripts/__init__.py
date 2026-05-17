# hf-gpu-cluster-optimizer `scripts/` package.
#
# Holds infra/deploy-side tooling: preflight validators, paid-launch
# wrappers, post-run sync helpers. Each script is a thin Python module
# that the PowerShell wrappers shell out to so the cost-control logic
# lives in one testable place.
