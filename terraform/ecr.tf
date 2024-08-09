# ecr.tf 
#create a public repository

resource "aws_ecrpublic_repository" "app" {
    repository_name = "my-flask-app-repo" 
}
