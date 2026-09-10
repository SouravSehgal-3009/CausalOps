variable "project_id" {
  type        = string
  description = "GCP project that owns the replay control-plane resources."
}

variable "region" {
  type        = string
  description = "GCP region for the private artifact bucket."
}

variable "artifact_bucket_name" {
  type        = string
  description = "Globally unique, private bucket name for finalized reports."
}

# Cloud Run API deployment (optional)

variable "api_image" {
  type        = string
  default     = ""
  description = "Fully-qualified Artifact Registry image (region-docker.pkg.dev/PROJECT/REPO/causalops-api:TAG) built from infra/docker/api.Dockerfile. Empty skips creating the Cloud Run service entirely -- this is a no-op until an image has actually been pushed."
}

variable "google_client_id" {
  type        = string
  default     = ""
  description = "Same Google OAuth client id the VM's CAUSALOPS_GOOGLE_CLIENT_ID already uses -- the Cloud Run API enforces the identical owner-allowlist auth, not a separate policy."
}

variable "allowed_owners" {
  type        = list(string)
  default     = []
  description = "Same owner allowlist as the VM's CAUSALOPS_ALLOWED_OWNERS, as a list here instead of a comma-joined string."
}

variable "billing_account_id" {
  type        = string
  default     = ""
  description = "Billing account id (billingAccounts/XXXXXX-XXXXXX-XXXXXX) for the google_billing_budget sign-off gate. Empty skips creating a budget -- managing billing budgets needs billing-account-level IAM the project-scoped terraform identity may not hold; see infra/DEPLOYMENT.md's Gotchas for how to apply it separately if so."
}

variable "budget_monthly_usd" {
  type        = number
  default     = 20
  description = "Conservative monthly notification threshold for the Phase D budget alert. Not a hard spend cap -- GCP has no per-request pre-spend gate the way cost_ledger.py enforces for Claude spend; this only emails/notifies once crossed."
}
