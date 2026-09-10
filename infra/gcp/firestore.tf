# `FirestoreCheckpointSaver`/`FirestoreReplayControlPlane` both resolve
# `firestore.Client()` against Application Default Credentials directly --
# unlike `GcsArtifactStore`, neither impersonates `replay_control_plane`.
# The VM's default compute SA (widened to the `cloud-platform` OAuth scope
# during setup, see infra/DEPLOYMENT.md) is therefore granted
# `roles/datastore.user` directly here, project-wide -- Firestore has no
# per-collection IAM condition equivalent to storage.tf's
# `resource.name.startsWith(...)` restriction, so this is coarser than the
# GCS grant by necessity, not by choice.
resource "google_project_service" "firestore" {
  project            = var.project_id
  service            = "firestore.googleapis.com"
  disable_on_destroy = false
}

resource "google_firestore_database" "default" {
  project     = var.project_id
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"

  depends_on = [google_project_service.firestore]
}

resource "google_project_iam_member" "vm_firestore_user" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${data.google_project.current.number}-compute@developer.gserviceaccount.com"

  depends_on = [google_project_service.firestore]
}
