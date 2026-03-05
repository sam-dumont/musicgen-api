output "api_key" {
  description = "Generated API key for authenticating with the musicgen-api"
  value       = random_password.api_key.result
  sensitive   = true
}

output "namespace" {
  description = "Kubernetes namespace where musicgen-api is deployed"
  value       = kubernetes_namespace_v1.musicgen.metadata[0].name
}

output "service_name" {
  description = "Name of the Kubernetes service"
  value       = kubernetes_service_v1.musicgen.metadata[0].name
}

output "service_port" {
  description = "Port the service listens on"
  value       = 8000
}
