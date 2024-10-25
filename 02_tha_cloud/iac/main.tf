terraform {
  required_providers {
    digitalocean = {
      source = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }
}

provider "digitalocean" {
  token = var.do_token
}

# Variables
variable "do_token" {
  type = string
}

variable "region" {
  default = "tor1"
}

resource "digitalocean_project" "wiki_cluster" {
  name        = "wiki_cluster"
  description = "Development infrastructure for SATMA-RAG"
  purpose     = "Development of SATMA-RAG"
  environment = "Development"
  resources   = [digitalocean_droplet.wiki.urn]
}

# Create a VPC for the private network
resource "digitalocean_vpc" "wiki_cluster_network" {
  name     = "wiki-cluster-network"
  region   = var.region
  ip_range = "10.10.0.0/16"
}

locals {
  cloud_init = <<-EOF
    #cloud-config
    users:
      - name: satma
        gecos: SATMA User
        groups: sudo
        shell: /bin/bash
        sudo: ALL=(ALL:ALL) ALL
        lock_passwd: false
        passwd: $(echo 'satmapoc' | openssl passwd -1 -stdin)
    chpasswd:
      list: |
        root:changeme
      expire: False
    runcmd:
      - sed -i 's/^#ClientAliveInterval.*/ClientAliveInterval 60/' /etc/ssh/sshd_config
      - systemctl restart ssh
      - apt-get update; apt-get install -y net-tools virtualenv python3-pip
      - cd /root; git clone https://github.com/kittisak-sajjapongse/llamaindex.git
  EOF
}

# Wiki & RAG server: public + private network for SSH access
resource "digitalocean_droplet" "wiki" {
  name   = "SATMA-Wiki"
  region = var.region
  # c-2  - 2 vCPUs, 4GB RAM
  # c-4  - 4 vCPUs, 8GB RAM
  # c-8  - 8 vCPUs, 16GB RAM
  # c-16 - 16 vCPUs, 32GB RAM
  # c-32 - 32 vCPUs, 64GB RAM
  size   = "c-60-intel"
  image  = "ubuntu-24-04-x64"

  # Connect to both public and private networks
  vpc_uuid = digitalocean_vpc.wiki_cluster_network.id
  ipv6 = false
  
  # Enable a public IP for external access
  # ssh_keys = ["your_ssh_key_id"]  # Add your SSH key ID here

  # Use cloud-init to change the root password on wiki server too
  user_data = local.cloud_init
}

# Output the public IP of the wiki server
output "wiki_public_ip" {
  value = digitalocean_droplet.wiki.ipv4_address
}

