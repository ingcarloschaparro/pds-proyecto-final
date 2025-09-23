resource "aws_security_group" "app_lb_sg" {
  name        = "${var.app_name}-alb-sg"
  description = "Security group for Application Load Balancer"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  /*ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }*/

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Application = var.app_name
    Environment = "production"
  }
}

resource "aws_lb" "app_lb" {
  name               = "${var.app_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.app_lb_sg.id]
  subnets            = data.aws_subnets.all.ids

  enable_deletion_protection = false

  tags = {
    Environment = "production"
    Application = var.app_name
  }
}

resource "aws_lb_target_group" "app_lb_tg_1" {
  name        = "${var.app_name}-app-tg"
  port        = var.app_port
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"

  health_check {
    enabled             = true
    path                = "/api/v1/health"
    protocol            = "HTTP"
    port                = "traffic-port"
    healthy_threshold   = 3
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    matcher             = "200-299"
  }

  tags = {
    Environment = "production"
    Application = var.app_name
  }

  depends_on = [aws_lb.app_lb]
}

resource "aws_lb_target_group" "app_lb_tg_2" {
  name        = "${var.app_name}-dashboard-tg"
  port        = var.dashboard_port
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"

  tags = {
    Environment = "production"
    Application = var.app_name
  }

  depends_on = [aws_lb.app_lb]
}

resource "aws_lb_listener" "app_lb_lis_http_1" {
  load_balancer_arn = aws_lb.app_lb.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app_lb_tg_1.arn
  }

  tags = {
    Environment = "production"
    Application = var.app_name
  }
}

resource "aws_lb_listener" "app_lb_lis_http_2" {
  load_balancer_arn = aws_lb.app_lb.arn
  port              = 8501
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app_lb_tg_2.arn
  }

  tags = {
    Environment = "production"
    Application = var.app_name
  }
}

/*resource "aws_lb_listener" "app_lb_lis_https" {
  count             = 1
  load_balancer_arn = aws_lb.app_lb.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-2016-08"
  #certificate_arn   = var.acm_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app_lb_tg.arn
  }

  tags = {
    Environment = "production"
    Application = var.app_name
  }
}*/

resource "aws_ecs_cluster" "main" {
  name = var.ecs_cluster_name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Environment = "production"
    Application = var.app_name
  }
}

resource "aws_ecs_task_definition" "main" {
  family                   = "${var.app_name}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 2048
  memory                   = 4096
  execution_role_arn       = local.lab_role_arn
  task_role_arn            = local.lab_role_arn

  ephemeral_storage {
    size_in_gib = 30  # Mínimo 20 GB, máximo 200 GB
  }

  container_definitions = jsonencode([{
    name      = var.app_name
    image     = "${var.ecr_repository_name}:${var.ecr_repository_version}"
    essential = true
    portMappings = [{
      containerPort = var.app_port
      hostPort      = var.app_port
      protocol      = "tcp"
    },
    {
      containerPort = var.dashboard_port
      hostPort      = var.dashboard_port
      protocol      = "tcp"
    }]

    # Configuración importante para contenedores grandes
    readonlyRootFilesystem = false  # Permitir escritura si es necesario

    # Resource limits
    memoryReservation = 3072  # 3 GB reservados
    cpu               = 1024  # 1 vCPU

    /*mountPoints = [
      {
        sourceVolume  = "app-data"
        containerPath = "/app/data"
        readOnly      = false
      }
    ]*/

     healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:${var.app_port}/api/v1/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }

    environment = [
      {

      }
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.ecs.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "ecs"
      }
    }
  }])

  /*volume {
    name = "app-data"

    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.app.id
      root_directory     = "/"
      transit_encryption = "ENABLED"
    }
  }*/

  tags = {
    Environment = "production"
    Application = var.app_name
  }
}

/*resource "aws_iam_role" "ecs_task_execution_role" {
  name = "ecs-task-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-01-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}*/

/*resource "aws_iam_role_policy_attachment" "ecs_task_execution_role" {
  role       = aws_iam_role.ecs_task_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}*/

# IAM Role para la task
/*resource "aws_iam_role" "ecs_task_role" {
  name = "ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-01-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}*/

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.app_name}"
  retention_in_days = 30

  tags = {
    Application = var.app_name
    Environment = "production"
  }
}

resource "aws_security_group" "ecs" {
  name        = "ecs-${var.app_name}-sg"
  description = "Security group for ECS service"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = var.app_port
    to_port     = var.app_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = var.dashboard_port
    to_port     = var.dashboard_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Application = var.app_name
    Environment = "production"
  }
}

resource "aws_ecs_service" "main" {
  name            = "${var.app_name}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.main.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.all.ids
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app_lb_tg_1.arn
    container_name   = var.app_name
    container_port   = var.app_port
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app_lb_tg_2.arn
    container_name   = var.app_name
    container_port   = var.dashboard_port
  }

  health_check_grace_period_seconds = 300

  deployment_controller {
    type = "ECS"
  }

  deployment_maximum_percent         = 200
  deployment_minimum_healthy_percent = 100

  # Force new deployment cuando cambia la task definition
  triggers = {
    task_definition = aws_ecs_task_definition.main.revision
  }

  depends_on = [
    aws_lb_listener.app_lb_lis_http_1,
    aws_lb_listener.app_lb_lis_http_2,
    aws_lb_target_group.app_lb_tg_1,
    aws_lb_target_group.app_lb_tg_2
  ]

  tags = {
    Environment = "production"
    Application = var.app_name
  }

  /*depends_on = [aws_iam_role_policy_attachment.ecs_task_execution_role]*/
}