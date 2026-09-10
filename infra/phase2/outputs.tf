output "control_plane_service_account_email" {
  value       = google_service_account.replay_control_plane.email
  description = "Attach this identity to the replay control-plane deployment, or (when attaching directly is not done) set it as CAUSALOPS_ARTIFACT_SERVICE_ACCOUNT so the app impersonates it for GCS access -- see vm_impersonates_control_plane."
}

output "artifact_bucket_name" {
  value       = google_storage_bucket.replay_artifacts.name
  description = "Private, versioned bucket for write-once investigation artifacts."
}

output "firestore_database_name" {
  value       = google_firestore_database.default.name
  description = "Firestore Native database backing FirestoreCheckpointSaver/FirestoreReplayControlPlane."
}

output "api_image_repository" {
  value       = google_artifact_registry_repository.api_images.name
  description = "Push infra/phaseD/Dockerfile's built image here, then set api_image to deploy the Cloud Run service."
}

output "cloud_run_api_url" {
  value       = length(google_cloud_run_v2_service.api) > 0 ? google_cloud_run_v2_service.api[0].uri : null
  description = "Live URL once var.api_image is set and applied; null while Phase D's Cloud Run service is skipped."
}
