# variables.tf

variable "region" {
  description = "The AWS region to deploy resources in"
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "The CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidr" {
  description = "The CIDR block for the public subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "container_image" {
  description = "The Docker image URI for the ECS container"
  type        = string
  #default     = "your-account-id.dkr.ecr.your-region.amazonaws.com/my-flask-app:latest"
  default     = "hello-world"

}

variable "ecs_task_memory" {
  description = "The amount of memory (in MiB) allocated to the ECS task"
  type        = number
  default     = 512
}

variable "ecs_task_cpu" {
  description = "The amount of CPU units allocated to the ECS task"
  type        = number
  default     = 256
}

variable "ecs_cluster_name" {
  description = "The name of the ECS cluster"
  type        = string
  default     = "my-ecs-cluster"
}

variable "ecs_task_family" {
  description = "The family name of the ECS task definition"
  type        = string
  default     = "flask-task"
}

variable "desired_task_count" {
  description = "The desired number of ECS tasks"
  type        = number
  default     = 1
}


variable "container_cpu" {
  description = "The number of CPU units reserved for the container"
  type        = number
  default = 256
}

variable "container_memory" {
  description = "The amount of memory (in MiB) reserved for the container"
  type        = number
  default = 512
}