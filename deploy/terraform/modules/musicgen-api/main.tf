# Random API key for authentication
resource "random_password" "api_key" {
  length  = 32
  special = false
}

# Namespace
resource "kubernetes_namespace_v1" "musicgen" {
  metadata {
    name = var.namespace
  }
}

# Secret for API key authentication
resource "kubernetes_secret_v1" "musicgen_api_key" {
  metadata {
    name      = "musicgen-api-key"
    namespace = kubernetes_namespace_v1.musicgen.metadata[0].name
  }

  data = {
    "api-key" = random_password.api_key.result
  }

  type = "kubernetes.io/opaque"
}

# PVC for model cache (MusicGen + Demucs models)
resource "kubernetes_persistent_volume_claim_v1" "musicgen_models" {
  metadata {
    name      = "musicgen-models"
    namespace = kubernetes_namespace_v1.musicgen.metadata[0].name
  }

  spec {
    access_modes       = ["ReadWriteOnce"]
    storage_class_name = var.storage_class != "" ? var.storage_class : null
    resources {
      requests = {
        storage = var.model_cache_size
      }
    }
  }

  wait_until_bound = false
}

# PVC for input/output data
resource "kubernetes_persistent_volume_claim_v1" "musicgen_data" {
  metadata {
    name      = "musicgen-data"
    namespace = kubernetes_namespace_v1.musicgen.metadata[0].name
  }

  spec {
    access_modes       = ["ReadWriteOnce"]
    storage_class_name = var.storage_class != "" ? var.storage_class : null
    resources {
      requests = {
        storage = var.data_size
      }
    }
  }

  wait_until_bound = false
}

# Deployment
resource "kubernetes_deployment_v1" "musicgen" {
  metadata {
    name      = "musicgen"
    namespace = kubernetes_namespace_v1.musicgen.metadata[0].name
    labels = {
      name = "musicgen"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        name = "musicgen"
      }
    }

    strategy {
      type = "Recreate"
    }

    template {
      metadata {
        labels = {
          name = "musicgen"
        }
      }

      spec {
        security_context {
          run_as_user  = 1001
          run_as_group = 1001
          fs_group     = 1001
        }

        node_selector = var.node_selector

        container {
          image             = var.image
          image_pull_policy = "Always"
          name              = "musicgen"

          resources {
            limits = merge(
              {
                cpu    = var.cpu_limit
                memory = var.memory_limit
              },
              var.gpu_enabled ? { "nvidia.com/gpu" = var.gpu_count } : {}
            )
            requests = merge(
              {
                cpu    = var.cpu_request
                memory = var.memory_request
              },
              var.gpu_enabled ? { "nvidia.com/gpu" = var.gpu_count } : {}
            )
          }

          # Environment variables
          env {
            name = "API_KEY"
            value_from {
              secret_key_ref {
                name = kubernetes_secret_v1.musicgen_api_key.metadata[0].name
                key  = "api-key"
              }
            }
          }

          env {
            name  = "OUTPUT_DIR"
            value = "/data/output"
          }

          env {
            name  = "PYTORCH_CUDA_ALLOC_CONF"
            value = "expandable_segments:True"
          }

          env {
            name  = "NVIDIA_VISIBLE_DEVICES"
            value = "all"
          }

          env {
            name  = "NVIDIA_DRIVER_CAPABILITIES"
            value = "all"
          }

          env {
            name  = "MUSICGEN_MODEL"
            value = var.musicgen_model
          }

          env {
            name  = "USE_STEM_AWARE_CROSSFADE"
            value = tostring(var.use_stem_aware_crossfade)
          }

          env {
            name  = "USE_QUALITY_LOOP"
            value = tostring(var.use_quality_loop)
          }

          env {
            name  = "MAX_REGEN_ATTEMPTS"
            value = tostring(var.max_regen_attempts)
          }

          # Volume mounts
          volume_mount {
            name       = "models"
            mount_path = "/home/appuser/.cache"
          }

          volume_mount {
            name       = "data"
            mount_path = "/data"
          }

          # Probes
          port {
            container_port = 8000
          }

          liveness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 60
            period_seconds        = 30
            timeout_seconds       = 10
            failure_threshold     = 3
          }

          readiness_probe {
            http_get {
              path = "/health"
              port = 8000
            }
            initial_delay_seconds = 30
            period_seconds        = 10
            timeout_seconds       = 5
            failure_threshold     = 3
          }
        }

        # Volumes
        volume {
          name = "models"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim_v1.musicgen_models.metadata[0].name
          }
        }

        volume {
          name = "data"
          persistent_volume_claim {
            claim_name = kubernetes_persistent_volume_claim_v1.musicgen_data.metadata[0].name
          }
        }
      }
    }
  }
}

# Service
resource "kubernetes_service_v1" "musicgen" {
  metadata {
    name      = "musicgen"
    namespace = kubernetes_namespace_v1.musicgen.metadata[0].name
    labels = {
      name = "musicgen"
    }
  }

  spec {
    selector = {
      name = "musicgen"
    }

    port {
      name        = "http"
      port        = 8000
      target_port = 8000
    }

    type = "ClusterIP"
  }
}
