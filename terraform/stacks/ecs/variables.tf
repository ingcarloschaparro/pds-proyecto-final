variable "region" {
  description = "AWS Region where the objects will be deployed."
  type        = string
  nullable    = false
}

variable "owner" {
  description = "The username of the author."
  type        = string
  nullable    = false
}

variable "ecr_repository_name" {
  description = "The name of the repositories in the Amazon ECR service."
  type        = string
  nullable    = false
}

variable "ecr_repository_version" {
  description = "The version of the repositories in the Amazon ECR service."
  type        = string
  nullable    = false
}

variable "ecs_cluster_name" {
  description = "Cluster ECS name"
  type        = string
  nullable    = false
}

variable "app_name" {
  description = "App name"
  type        = string
  nullable    = false
}

variable "app_port" {
  description = "App port"
  type        = number
  nullable    = false
}










