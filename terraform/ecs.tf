resource "aws_ecs_cluster" "main" {
  name = "my-ecs-cluster"
}

resource "aws_ecs_task_definition" "flask" {
  family                = "flask-task"
  network_mode          = "awsvpc"  # Ensure this is set to 'awsvpc'
  requires_compatibilities = ["FARGATE"]  # Use FARGATE if you're using Fargate launch type
  cpu                   = "256"  # Set required CPU units
  memory                = "512"  # Set required memory in MiB

  container_definitions = jsonencode([{
    name      = "flask-container"
    image     = var.container_image
    essential = true
    memory    = var.ecs_task_memory
    cpu       = var.ecs_task_cpu
    portMappings = [
      {
        containerPort = 5000
        hostPort      = 5000
      }
    ]
  }])

  #execution_role_arn = var.execution_role_arn  # Ensure these are defined
  #task_role_arn      = var.task_role_arn
}

resource "aws_ecs_service" "flask" {
  name            = "flask-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.flask.arn
  desired_count   = 1
  launch_type     = "FARGATE"  # Ensure this matches your requirements

  network_configuration {
    subnets          = aws_subnet.public[*].id
    assign_public_ip = false  # Valid for 'awsvpc' network mode
    security_groups   = [aws_security_group.ecs.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.flask.arn
    container_name   = "flask-container"
    container_port   = 5000
  }

  depends_on = [aws_lb_listener.http]
}
