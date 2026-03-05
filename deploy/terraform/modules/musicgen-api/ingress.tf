# Ingress - only created when var.domain is set
resource "kubernetes_ingress_v1" "musicgen" {
  count = var.domain != null ? 1 : 0

  metadata {
    name      = "musicgen"
    namespace = kubernetes_namespace_v1.musicgen.metadata[0].name
    annotations = var.tls_enabled && var.tls_issuer != "" ? {
      "cert-manager.io/cluster-issuer" = var.tls_issuer
    } : {}
  }

  spec {
    ingress_class_name = var.ingress_class

    dynamic "tls" {
      for_each = var.tls_enabled ? [1] : []
      content {
        hosts       = [var.domain]
        secret_name = "musicgen-tls"
      }
    }

    rule {
      host = var.domain
      http {
        path {
          path      = "/"
          path_type = "Prefix"
          backend {
            service {
              name = kubernetes_service_v1.musicgen.metadata[0].name
              port {
                number = 8000
              }
            }
          }
        }
      }
    }
  }
}
