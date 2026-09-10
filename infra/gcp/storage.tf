# Required for the VM's default service account to impersonate
# `replay_control_plane` at all (`google_service_account_iam_member.
# vm_impersonates_control_plane` below) -- without it, every impersonated
# credential request is refused with SERVICE_DISABLED, discovered live
# during validation.
resource "google_project_service" "iam_credentials" {
  project            = var.project_id
  service            = "iamcredentials.googleapis.com"
  disable_on_destroy = false
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

# The VM's own runtime identity is its default Compute Engine service
# account, not `replay_control_plane` -- no `google_service_account_key`
# exists (see the bucket resource's own comment: no key ever created), so
# the application impersonates `replay_control_plane` for exactly as long
# as one GCS request instead. This is the only grant the VM's default SA
# receives; it still has no direct bucket role of its own.
resource "google_service_account_iam_member" "vm_impersonates_control_plane" {
  service_account_id = google_service_account.replay_control_plane.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "serviceAccount:${data.google_project.current.number}-compute@developer.gserviceaccount.com"

  depends_on = [google_project_service.iam_credentials]
}
