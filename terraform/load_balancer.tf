resource "aws_lb" "flask" {
  name               = "flask-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
  enable_deletion_protection = false
  idle_timeout       = 400
}

resource "aws_lb_target_group" "flask" {
  name     = "flask-target-group"
  port     = 5000
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/"  # Ensure this path is correctly defined in your application
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 2
    protocol            = "HTTP"
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.flask.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.flask.arn
  }
}
