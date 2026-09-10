# Cloud Run hosts the owner-facing HTTP API only -- the replay worker (lab
# access, MCP spawning) stays VM-hosted, per infra/DEPLOYMENT.md's own
# scoping notes (lab/docker-compose.yml binds services to 127.0.0.1 and
# shares a bind-mounted runs/ directory with the worker; neither survives a
# stateless container). Both the Cloud Run API and the VM worker must point
# at the same control-plane backend once this exists -- Cloud Run has no
# local disk to share a SQLite file with the VM -- so this only makes sense
# together with CAUSALOPS_CONTROL_PLANE_BACKEND=firestore on both sides.
resource "google_project_service" "run" {
  project            = var.project_id
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

# Dedicated identity for the Cloud Run revision -- deliberately not
# `replay_control_plane` (that one is scoped to GCS artifact writes/reads
# only) and not the VM's default compute SA (that one is scoped to
# Firestore + GCS-impersonation-granting only). Cloud Run needs Firestore
# access for the same reason the VM worker does; it needs no GCS role at
# all -- the owner-facing API never reads/writes the artifact bucket
# directly (report content is served out of the Firestore document,
# written there once by the VM worker's own `finalize()`).
resource "google_service_account" "cloud_run_api" {
  account_id   = "causalops-api"
  display_name = "CausalOps Cloud Run API"
}

resource "google_project_iam_member" "cloud_run_firestore_user" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.cloud_run_api.email}"

  depends_on = [google_project_service.firestore]
}

# Empty var.api_image means "no image has been pushed yet" -- this whole
# resource (and the public-invoker grant below) is skipped rather than
# applying with a placeholder that would immediately fail to start.
resource "google_cloud_run_v2_service" "api" {
  count    = var.api_image == "" ? 0 : 1
  project  = var.project_id
  name     = "causalops-api"
  location = var.region

  template {
    service_account = google_service_account.cloud_run_api.email

    containers {
      image = var.api_image

      env {
        name  = "CAUSALOPS_ALLOWED_OWNERS"
        value = join(",", var.allowed_owners)
      }
      env {
        name  = "CAUSALOPS_GOOGLE_CLIENT_ID"
        value = var.google_client_id
      }
      env {
        name  = "CAUSALOPS_PROJECT_ROOT"
        value = "/app"
      }
      env {
        name  = "CAUSALOPS_CONTROL_PLANE_BACKEND"
        value = "firestore"
      }
      env {
        # Cloud Run has no lab/docker access -- without this, app()'s own
        # BackgroundControlPlaneWorkers would start here too and race the
        # VM's real worker for jobs it could never actually run (found
        # live this session as a real gap in the first Cloud Run deploy).
        name  = "CAUSALOPS_RUN_WORKER"
        value = "false"
      }

      ports {
        container_port = 8080
      }
    }
  }

  depends_on = [
    google_project_service.run,
    google_firestore_database.default,
    google_project_iam_member.cloud_run_firestore_user,
  ]
}

# The dashboard/API is meant to be internet-reachable; Google ID token auth
# is still enforced at the application layer (GoogleIdentityVerifier +
# CAUSALOPS_ALLOWED_OWNERS) exactly as it is on the VM today -- this grant
# only controls who can reach the Cloud Run *transport*, not who the
# application accepts as an owner.
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  count    = var.api_image == "" ? 0 : 1
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api[0].name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
