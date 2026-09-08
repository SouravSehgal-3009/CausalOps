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
