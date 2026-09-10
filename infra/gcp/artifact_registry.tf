resource "google_project_service" "artifact_registry" {
  project            = var.project_id
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "api_images" {
  project       = var.project_id
  location      = var.region
  repository_id = "causalops-api"
  format        = "DOCKER"

  depends_on = [google_project_service.artifact_registry]
}

# The VM's default compute SA is what actually runs `docker push` (the
# gcloud CLI's active account, not terraform's own broader ADC identity) --
# without this it can create the repo but not push into it
# (artifactregistry.repositories.uploadArtifacts denied, confirmed live).
resource "google_artifact_registry_repository_iam_member" "vm_pushes_api_images" {
  project    = var.project_id
  location   = google_artifact_registry_repository.api_images.location
  repository = google_artifact_registry_repository.api_images.repository_id
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${data.google_project.current.number}-compute@developer.gserviceaccount.com"
}
