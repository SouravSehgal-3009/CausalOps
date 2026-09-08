terraform {
  required_version = ">= 1.8"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# This identity is for the replay control plane only. It has no owner role,
# service-account-key, compute, network, or provider-management permissions.
resource "google_service_account" "replay_control_plane" {
  account_id   = "causalops-replay-control"
  display_name = "CausalOps replay control plane"
}

resource "google_project_iam_member" "control_plane_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.replay_control_plane.email}"
}

# Final reports are versioned, private objects. Application code still enforces
# write-once finalization before it hands an artifact to the delivery outbox.
resource "google_storage_bucket" "replay_artifacts" {
  name                        = var.artifact_bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false

  versioning {
    enabled = true
  }
}

locals {
  artifact_prefix = "projects/_/buckets/${google_storage_bucket.replay_artifacts.name}/objects/investigations/"
}

# Restrict both permissions to final-investigation objects. The API reads an
# exact known report path; it does not require bucket-wide access.
resource "google_storage_bucket_iam_member" "control_plane_artifact_writer" {
  bucket = google_storage_bucket.replay_artifacts.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.replay_control_plane.email}"

  condition {
    title       = "write_final_investigation_artifacts"
    description = "Create artifacts below the investigation prefix only."
    expression  = "resource.name.startsWith('${local.artifact_prefix}')"
  }
}

resource "google_storage_bucket_iam_member" "control_plane_artifact_reader" {
  bucket = google_storage_bucket.replay_artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.replay_control_plane.email}"

  condition {
    title       = "read_final_investigation_artifacts"
    description = "Read artifacts below the investigation prefix only."
    expression  = "resource.name.startsWith('${local.artifact_prefix}')"
  }
}
