resource "aws_ecr_repository" "main" {
  for_each = { for name in var.repository_name : name => name }
  
  name                 = "${each.key}" 
  image_tag_mutability = "MUTABLE" 
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }
}

resource "aws_ecr_lifecycle_policy" "lifecycle_policy" {
  for_each = { for name in var.repository_name : name => name }
  
  repository = aws_ecr_repository.main[each.key].name 
  policy     = local.policy_document 
}