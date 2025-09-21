module "ecr_repository" {
  source = "../../modules/container_registry"
  keep_tags_number = var.keep_tags_number
  repository_name  = var.repository_name
}