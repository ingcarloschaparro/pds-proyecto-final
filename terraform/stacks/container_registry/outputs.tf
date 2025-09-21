output "repositories" {
  value       =  { for k, rep in module.ecr_repository : k => rep }
  description = "Created repositories."
}