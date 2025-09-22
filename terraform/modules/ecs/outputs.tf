output "repository_name" { 
  value       = { for k, rep in aws_ecr_repository.main : k => rep.name }
  description = "Repository Name."
}

output "repository_arn" { 
  value       = { for k, rep in aws_ecr_repository.main : k => rep.arn }
  description = "Repository ARN."
}