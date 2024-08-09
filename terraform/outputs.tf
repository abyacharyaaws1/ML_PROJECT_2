output "vpc_id" {
  value = aws_vpc.main.id
}

output "public_subnet_ids" {
  value = aws_subnet.public[*].id
}

output "ecs_cluster_id" {
  value = aws_ecs_cluster.main.id
}

output "ecr_repository_uri" {
  value = aws_ecrpublic_repository.app.repository_uri
}

output "ecs_task_definition_arn" {
  value = aws_ecs_task_definition.flask.arn
}
