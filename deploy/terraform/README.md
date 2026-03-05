# musicgen-api Terraform Module

Deploys musicgen-api on any Kubernetes cluster with optional GPU support and ingress.

## Usage

```hcl
module "musicgen" {
  source = "./modules/musicgen-api"

  musicgen_model = "facebook/musicgen-medium"
  gpu_enabled    = true
  domain         = "musicgen.example.com"
  ingress_class  = "nginx"
  tls_issuer     = "letsencrypt-prod"
  storage_class  = "standard"

  node_selector = {
    "nvidia.com/gpu.present" = "true"
  }
}
```

## Inputs

| Name | Description | Type | Default |
|------|-------------|------|---------|
| `namespace` | Kubernetes namespace to deploy into | `string` | `"musicgen"` |
| `image` | Container image for the musicgen-api | `string` | `"ghcr.io/sam-dumont/musicgen-api:latest"` |
| `musicgen_model` | MusicGen model to use | `string` | `"facebook/musicgen-small"` |
| `gpu_enabled` | Whether to request GPU resources | `bool` | `true` |
| `gpu_count` | Number of GPUs to request | `number` | `1` |
| `storage_class` | Storage class for PVCs (empty = cluster default) | `string` | `""` |
| `model_cache_size` | PVC size for model cache | `string` | `"15Gi"` |
| `data_size` | PVC size for input/output data | `string` | `"20Gi"` |
| `domain` | Domain for ingress (null = no ingress) | `string` | `null` |
| `ingress_class` | Ingress class name | `string` | `"nginx"` |
| `tls_enabled` | Enable TLS on ingress | `bool` | `true` |
| `tls_issuer` | cert-manager ClusterIssuer name | `string` | `""` |
| `use_stem_aware_crossfade` | Enable Demucs stem-aware transitions | `bool` | `false` |
| `use_quality_loop` | Enable quality-based regeneration | `bool` | `true` |
| `max_regen_attempts` | Max regeneration attempts | `number` | `3` |
| `cpu_request` | CPU request | `string` | `"2"` |
| `cpu_limit` | CPU limit | `string` | `"4"` |
| `memory_request` | Memory request | `string` | `"6Gi"` |
| `memory_limit` | Memory limit | `string` | `"10Gi"` |
| `node_selector` | Node selector labels | `map(string)` | `{}` |

## Outputs

| Name | Description |
|------|-------------|
| `api_key` | Generated API key (sensitive) |
| `namespace` | Namespace name |
| `service_name` | Service name |
| `service_port` | Service port (8000) |

## Notes

**GPU support:** When `gpu_enabled = true`, the deployment requests `nvidia.com/gpu` resources. Your cluster needs the NVIDIA device plugin installed. For CPU-only, set `gpu_enabled = false`: inference will be slower but functional.

**Storage class:** Leave `storage_class` empty to use your cluster's default. Set it explicitly if you need a specific provisioner (e.g. `gp3` on EKS, `standard` on GKE, `do-block-storage` on DigitalOcean).

**Ingress:** Only created when `domain` is set. The module supports any ingress controller via `ingress_class`. TLS is handled by cert-manager when `tls_issuer` is provided. If you manage TLS outside of cert-manager, set `tls_enabled = true` and `tls_issuer = ""`: you'll need to create the `musicgen-tls` secret yourself.

**Retrieving the API key:**

```bash
terraform output -raw api_key
```
