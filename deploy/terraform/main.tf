provider "kubernetes" {
  # Configure your cluster connection
  # config_path = "~/.kube/config"
}

module "musicgen" {
  source = "./modules/musicgen-api"

  musicgen_model = "facebook/musicgen-medium"
  gpu_enabled    = true
  domain         = "musicgen.example.com"
  ingress_class  = "nginx"
  tls_issuer     = "letsencrypt-prod"
  storage_class  = "standard"
}

output "api_key" {
  value     = module.musicgen.api_key
  sensitive = true
}

output "namespace" {
  value = module.musicgen.namespace
}
