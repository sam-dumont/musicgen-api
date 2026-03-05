variable "namespace" {
  description = "Kubernetes namespace to deploy into"
  type        = string
  default     = "musicgen"
}

variable "image" {
  description = "Container image for the musicgen-api"
  type        = string
  default     = "ghcr.io/sam-dumont/musicgen-api:latest"
}

variable "musicgen_model" {
  description = "MusicGen model to use (facebook/musicgen-small, facebook/musicgen-medium, etc.)"
  type        = string
  default     = "facebook/musicgen-small"
}

variable "gpu_enabled" {
  description = "Whether to request GPU resources for the deployment"
  type        = bool
  default     = true
}

variable "gpu_count" {
  description = "Number of GPUs to request when gpu_enabled is true"
  type        = number
  default     = 1
}

variable "storage_class" {
  description = "Kubernetes storage class for persistent volumes (empty string uses cluster default)"
  type        = string
  default     = ""
}

variable "model_cache_size" {
  description = "Size of the persistent volume for model cache (MusicGen + Demucs models)"
  type        = string
  default     = "15Gi"
}

variable "data_size" {
  description = "Size of the persistent volume for input/output data"
  type        = string
  default     = "20Gi"
}

variable "domain" {
  description = "Domain name for the ingress. No ingress is created when null."
  type        = string
  default     = null
}

variable "ingress_class" {
  description = "Ingress class name (e.g. nginx, traefik)"
  type        = string
  default     = "nginx"
}

variable "tls_enabled" {
  description = "Enable TLS on the ingress"
  type        = bool
  default     = true
}

variable "tls_issuer" {
  description = "cert-manager ClusterIssuer name for TLS certificates (empty string disables the annotation)"
  type        = string
  default     = ""
}

variable "use_stem_aware_crossfade" {
  description = "Enable Demucs-based stem-aware transitions"
  type        = bool
  default     = false
}

variable "use_quality_loop" {
  description = "Enable quality metrics with automatic regeneration for poor transitions"
  type        = bool
  default     = true
}

variable "max_regen_attempts" {
  description = "Maximum regeneration attempts when quality loop is enabled"
  type        = number
  default     = 3
}

variable "cpu_request" {
  description = "CPU request for the musicgen container"
  type        = string
  default     = "2"
}

variable "cpu_limit" {
  description = "CPU limit for the musicgen container"
  type        = string
  default     = "4"
}

variable "memory_request" {
  description = "Memory request for the musicgen container"
  type        = string
  default     = "6Gi"
}

variable "memory_limit" {
  description = "Memory limit for the musicgen container"
  type        = string
  default     = "10Gi"
}

variable "node_selector" {
  description = "Node selector labels for pod scheduling"
  type        = map(string)
  default     = {}
}
