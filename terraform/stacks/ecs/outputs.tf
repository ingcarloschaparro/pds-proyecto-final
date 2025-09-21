output "ecs_cluster_name" {
  description = "Nombre del cluster ECS"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Nombre del servicio ECS"
  value       = aws_ecs_service.main.name
}

output "task_definition_arn" {
  description = "ARN de la task definition"
  value       = aws_ecs_task_definition.main.arn
}

output "alb_dns_name" {
  description = "DNS name del Application Load Balancer"
  value       = aws_lb.app_lb.dns_name
}

output "alb_zone_id" {
  description = "Zone ID del ALB"
  value       = aws_lb.app_lb.zone_id
}

output "target_group_arn" {
  description = "ARN del target group"
  value       = aws_lb_target_group.app_lb_tg.arn
}

output "service_url" {
  description = "URL del servicio"
  value       = "http://${aws_lb.app_lb.dns_name}"
}

output "https_service_url" {
  description = "HTTPS URL del servicio (si está habilitado)"
  value       = "https://${aws_lb.app_lb.dns_name}"
}