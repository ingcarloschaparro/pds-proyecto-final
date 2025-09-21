variable "keep_tags_number" {
  description = "The number of image tags to retain in the registry."
  type        = number
}

variable "repository_name" {
  description = "The name of the repositories in the Amazon ECR service."
  type        = list(string)
  nullable    = false
}
