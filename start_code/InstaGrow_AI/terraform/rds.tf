terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

resource "aws_db_instance" "instagrow_rds" {
  identifier        = "instagrow-postgres"
  engine           = "postgres"
  engine_version    = "15.4"
  instance_class    = "db.t3.micro"
  allocated_storage = 20
  storage_type      = "gp2"
  
  db_name  = "instagrow_ai"
  username = var.db_username
  password = var.db_password
  
  db_subnet_group_name   = aws_db_subnet_group.default.name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  
  skip_final_snapshot = false
  final_snapshot_identifier = "instagrow-final-snapshot"
  
  backup_retention_period = 7
  multi_az               = false
  
  tags = {
    Name        = "InstaGrow-RDS"
    Environment = var.environment
  }
}

resource "aws_db_subnet_group" "default" {
  name       = "instagrow-db-subnet-group"
  subnet_ids = var.subnet_ids
  
  tags = {
    Name = "InstaGrow DB Subnet Group"
  }
}

resource "aws_security_group" "rds_sg" {
  name_prefix = "instagrow-rds-"
  vpc_id      = var.vpc_id
  
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = var.vpc_cidr_blocks
  }
  
  tags = {
    Name = "InstaGrow-RDS-SG"
  }
}

output "db_endpoint" {
  description = "RDS endpoint"
  value       = aws_db_instance.instagrow_rds.endpoint
}

output "db_port" {
  description = "RDS port"
  value       = aws_db_instance.instagrow_rds.port
}
