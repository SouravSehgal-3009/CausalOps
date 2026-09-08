output "control_plane_service_account_email" {
  value       = google_service_account.replay_control_plane.email
  description = "Attach this identity to the replay control-plane deployment."
}

output "artifact_bucket_name" {
  value       = google_storage_bucket.replay_artifacts.name
  description = "Private, versioned bucket for write-once investigation artifacts."
}
