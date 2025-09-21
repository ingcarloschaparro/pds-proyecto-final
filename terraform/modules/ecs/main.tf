module "ecs_cluster" {
  source  = "terraform-aws-modules/ecs/aws"

  cluster_name    = var.ecs_cluster_name

  cluster_configuration = {
    execute_command_configuration = {
      logging = "OVERRIDE"
      log_configuration = {
        cloud_watch_log_group_name = "/aws/ecs/aws-ec2"
      }
    }
  }

  default_capacity_provider_strategy = {
    FARGATE = {
      weight = 50
      base   = 20
    }
    FARGATE_SPOT = {
      weight = 50
    }
  }

  services = {
    pls-api = {
      cpu    = 1024
      memory = 4096

      # Container definition(s)
      container_definitions = {

        pls-app = {
          cpu       = 512
          memory    = 1024
          essential = true
          image     = var.ecr_repository_name
          portMappings = [
            {
              name          = var.app_name
              containerPort = var.app_port
              protocol      = "tcp"
            }
          ]

          readonlyRootFilesystem = false

          enable_cloudwatch_logging = false
          logConfiguration = {
            logDriver = "awsfirelens"
            options = {
              Name                    = "firehose"
              region                  = "eu-west-1"
              delivery_stream         = var.app_name
              log-driver-buffer-limit = "2097152"
            }
          }
          memoryReservation = 100
        }
      }

      service_connect_configuration = {
        namespace = "pls"
        service = [{
          client_alias = {
            port     = var.app_port
            dns_name = var.app_name
          }
          port_name      = "pls-port"
          discovery_name = var.app_name
        }]
      }

      load_balancer = {
        service = {
          target_group_arn = "arn:aws:elasticloadbalancing:eu-east-1:1234567890:targetgroup/bluegreentarget1/209a844cd01825a4"
          container_name   = var.app_name
          container_port   = 80
        }
      }

      subnet_ids = ["subnet-abcde012", "subnet-bcde012a", "subnet-fghi345a"]
      security_group_ingress_rules = {
        alb_3000 = {
          description                  = "Service port"
          from_port                    = var.app_port
          ip_protocol                  = "tcp"
          referenced_security_group_id = "sg-12345678"
        }
      }
      security_group_egress_rules = {
        all = {
          ip_protocol = "-1"
          cidr_ipv4   = "0.0.0.0/0"
        }
      }
    }
  }

  tags = {
    Environment = "Development"
    Project     = "Example"
  }
}