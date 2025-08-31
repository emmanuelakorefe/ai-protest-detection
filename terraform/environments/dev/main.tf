terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-2"
}

resource "aws_vpc" "campus_ai_dev" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name        = "campus-ai-dev"
    Environment = "development"
    Project     = "campus-ai"
  }
}

resource "aws_internet_gateway" "campus_ai_dev" {
  vpc_id = aws_vpc.campus_ai_dev.id
  
  tags = {
    Name = "campus-ai-dev-igw"
  }
}

resource "aws_subnet" "public_dev" {
  vpc_id                  = aws_vpc.campus_ai_dev.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "us-east-2a"
  map_public_ip_on_launch = true
  
  tags = {
    Name = "campus-ai-dev-public"
  }
}
# Route table for public subnet
resource "aws_route_table" "public_dev" {
  vpc_id = aws_vpc.campus_ai_dev.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.campus_ai_dev.id
  }

  tags = {
    Name = "campus-ai-dev-public-rt"
  }
}

resource "aws_route_table_association" "public_dev" {
  subnet_id      = aws_subnet.public_dev.id
  route_table_id = aws_route_table.public_dev.id
}

# Security group for your app
resource "aws_security_group" "campus_ai_app" {
  name        = "campus-ai-dev-app"
  description = "Security group for Campus AI application"
  vpc_id      = aws_vpc.campus_ai_dev.id

  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
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
    Name = "campus-ai-dev-app-sg"
  }
}
resource "aws_instance" "campus_ai_ec2" {
  ami                    = "ami-0d9a665f802ae6227" # Ubuntu 22.04 LTS for us-east-2
  instance_type          = "t2.micro"
  subnet_id              = aws_subnet.public_dev.id
  vpc_security_group_ids = [aws_security_group.campus_ai_app.id]
  key_name               = "rosefaith"

  associate_public_ip_address = true

  user_data = <<-EOF
              #!/bin/bash
              apt update -y
              apt install -y docker.io docker-compose git
              cd /home/ubuntu
              git clone https://github.com/masterwin1122/ai-protest-detection.git
              cd ai-protest-detection
              docker-compose up -d
              EOF

  tags = {
    Name = "campus-ai-ec2"
  }
}
