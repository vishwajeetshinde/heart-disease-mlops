# Kubernetes Deployment Guide

Complete guide for deploying the Heart Disease Prediction API to Kubernetes (local or cloud).

---

## 🎯 Deployment Options

| Platform | Difficulty | Use Case | Cost |
|----------|-----------|----------|------|
| **Docker Desktop K8s** | Easy | Local development | Free |
| **Minikube** | Easy | Local testing | Free |
| **GKE/EKS/AKS** | Medium | Production | Paid |

### Recommended: Docker Desktop Kubernetes
- Built-in, no extra installation
- Perfect for local testing
- Enable in: Settings → Kubernetes → Enable Kubernetes

---

## 📋 Prerequisites

### Required Tools

```bash
# 1. Docker Desktop (includes kubectl)
# Download: https://www.docker.com/products/docker-desktop/
# Enable Kubernetes in Settings

# 2. Verify kubectl
kubectl version --client

# 3. Install Helm (optional, for Helm deployment)
brew install helm
helm version

# 4. Verify cluster is running
kubectl cluster-info
kubectl get nodes
```

---

## 🚀 Deployment Steps

## Method A: Using kubectl (Direct Deployment)

### Step 1: Build and Load Docker Image

```bash
# Navigate to project directory
cd /Users/VSE18/Desktop/Projects/heart-disease-mlops

# Build Docker image
docker build -t heart-disease-api:latest .

# For Minikube, load image into cluster
minikube image load heart-disease-api:latest

# For Docker Desktop, image is already available
```

### Step 2: Deploy Using Script (Easiest)

```bash
# Full deployment (build + deploy + ingress + HPA)
./k8s/deploy.sh deploy-all
---

## 🚀 Quick Start (3 Steps)

```bash
# Step 1: Navigate to project
cd /Users/VSE18/Desktop/Projects/heart-disease-mlops

# Step 2: Deploy everything
./k8s/deploy.sh deploy-all

# Step 3: Access API
kubectl port-forward service/heart-disease-api-service 8000:80
# Open: http://localhost:8000/docs
```

---

## 📦 Method A: kubectl Deployment

### Full Automated Deployment

```bash
# Build Docker image and deploy everything
./k8s/deploy.sh deploy-all

# This runs:
# - Builds Docker image
# - Loads to K8s cluster
# - Deploys pods (3 replicas)
# - Creates service (LoadBalancer)
# - Configures ingress
# - Sets up autoscaling (HPA)
```

### Manual Step-by-Step

```bash
# 1. Build image
docker build -t heart-disease-api:latest .

# 2. Load to Minikube (skip for Docker Desktop)
minikube image load heart-disease-api:latest

# 3. Deploy resources
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service-nodeport.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# 4. Verify
kubectl get all -l app=heart-disease-api
```

### Access Deployed API

**Option 1: Port-forward (Works everywhere)**
```bash
kubectl port-forward service/heart-disease-api-service 8000:80
curl http://localhost:8000/health
```

**Option 2: NodePort (Docker Desktop)**
```bash
curl http://localhost:30080/health
```

**Option 3: Minikube Service**
```bash
minikube service heart-disease-api-service --url
```

**Option 4: Ingress**
```bash
# Add to /etc/hosts
echo "127.0.0.1 heart-disease-api.local" | sudo tee -a /etc/hosts

# Access
curl http://heart-disease-api.local/health
```elm/deploy-helm.sh install

# Check status
./helm/deploy-helm.sh status

# Get service URL
./helm/deploy-helm.sh url

# Test API
./helm/deploy-helm.sh test
```

### Step 3: Manual Helm Deployment (Alternative)

```bash
# Lint chart
helm lint ./helm/heart-disease-api

# Install with default values
helm install heart-disease-api ./helm/heart-disease-api

# Install with custom values
helm install heart-disease-api ./helm/heart-disease-api \
  --set replicaCount=5 \
  --set service.type=NodePort

# Install in custom namespace
helm install heart-disease-api ./helm/heart-disease-api \
  --namespace production \
  --create-namespace

# Upgrade release
helm upgrade heart-disease-api ./helm/heart-disease-api

# Check status
helm status heart-disease-api

# Get values
helm get values heart-disease-api

# Rollback to previous version
helm rollback heart-disease-api

# Uninstall
helm uninstall heart-disease-api
```

### Step 4: Customize Helm Values

Create `custom-values.yaml`:
```yaml
replicaCount: 5

resources:
  limits:
    cpu: 1000m
    memory: 1Gi
  requests:
    cpu: 500m
    memory: 512Mi

service:
  type: LoadBalancer

ingress:
  enabled: true
  hosts:
    - host: my-api.example.com
      paths:
        - path: /
          pathType: Prefix

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
```

Install with custom values:
```bash
helm install heart-disease-api ./helm/heart-disease-api \
  -f custom-values.yaml
```

---

## 🧪 Testing & Verification

### Automated Verification Script

```bash
# Run complete verification
python3 k8s/verify_deployment.py

# This checks:
# - Kubernetes resources status
# - Pod health
# - Service endpoints
# - API health endpoint
# - Prediction functionality
```

### Manual Testing

```bash
# 1. Check resources
kubectl get all -l app=heart-disease-api

# 2. Check pod logs
kubectl logs -l app=heart-disease-api --tail=50

# 3. Port-forward
kubectl port-forward service/heart-disease-api-service 8000:80

# 4. Test health
curl http://localhost:8000/health

# 5. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_input.json | jq

# 6. Open Swagger UI
open http://localhost:8000/docs
```

### Available Helper Scripts

```bash
# kubectl scripts
./k8s/deploy.sh prereq      # Check prerequisites
./k8s/deploy.sh build        # Build Docker image
./k8s/deploy.sh deploy       # Deploy to K8s
./k8s/deploy.sh status       # Check status
./k8s/deploy.sh url          # Get service URL
./k8s/deploy.sh logs         # View logs
./k8s/deploy.sh cleanup      # Delete all resources

# Helm scripts
./helm/deploy-helm.sh install    # Install chart
./helm/deploy-helm.sh status     # Show status
./helm/deploy-helm.sh test       # Test API
./helm/deploy-helm.sh upgrade    # Upgrade release
./helm/deploy-helm.sh uninstall  # Remove release
```

---

## 📊 Monitoring

### View Logs
```bash
# All pods
kubectl logs -l app=heart-disease-api --tail=100 -f

# Specific pod
kubectl logs <pod-name> -f
```

### Check Resources
```bash
# Pod status
kubectl get pods -o wide

# Resource usage (requires metrics-server)
kubectl top pods
kubectl top nodes

# HPA status
kubectl get hpa
kubectl describe hpa heart-disease-api-hpa
```

### Events
```bash
# Recent events
kubectl get events --sort-by=.metadata.creationTimestamp

# Pod-specific events
kubectl describe pod <pod-name>
```

---

## 🔧 Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl get pods -l app=heart-disease-api

# View pod details
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>

# Common issues:
# - ImagePullBackOff: Image not loaded (Minikube: minikube image load)
# - CrashLoopBackOff: Check logs for errors
# - Pending: Insufficient resources or scheduling issues
```

### Service Not Accessible

```bash
# Check service
kubectl get service heart-disease-api-service

# Check endpoints
kubectl get endpoints heart-disease-api-service

# Use port-forward as fallback
kubectl port-forward service/heart-disease-api-service 8000:80
```

### HPA Not Working

```bash
# Check metrics-server
kubectl get deployment metrics-server -n kube-system

# Install if missing
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# For Docker Desktop, may need to edit:
kubectl edit deployment metrics-server -n kube-system
# Add to args: --kubelet-insecure-tls
```

### Ingress Not Working

```bash
# Check ingress controller
kubectl get ingressclass

# Install NGINX ingress (Minikube)
minikube addons enable ingress

# Install NGINX ingress (Docker Desktop)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
```

---

## ☁️ Cloud Deployment

### Google Kubernetes Engine (GKE)

```bash
# Create cluster
gcloud container clusters create heart-disease-cluster \
  --num-nodes=3 \
  --machine-type=e2-standard-2 \
  --zone=us-central1-a

# Get credentials
gcloud container clusters get-credentials heart-disease-cluster

# Push image to GCR
docker tag heart-disease-api:latest gcr.io/PROJECT_ID/heart-disease-api:latest
docker push gcr.io/PROJECT_ID/heart-disease-api:latest

# Update k8s/deployment.yaml with GCR image path
# Deploy
./k8s/deploy.sh deploy-all
```

### Amazon EKS

```bash
# Create cluster
eksctl create cluster \
  --name heart-disease-cluster \
  --region us-west-2 \
  --nodes 3

# Push to ECR
aws ecr create-repository --repository-name heart-disease-api
docker tag heart-disease-api:latest ACCOUNT.dkr.ecr.REGION.amazonaws.com/heart-disease-api:latest
docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/heart-disease-api:latest

# Deploy
./k8s/deploy.sh deploy-all
```

### Azure AKS

```bash
# Create cluster
az aks create \
  --resource-group heart-disease-rg \
  --name heart-disease-cluster \
  --node-count 3

# Get credentials
az aks get-credentials --resource-group heart-disease-rg --name heart-disease-cluster

# Push to ACR
az acr create --resource-group heart-disease-rg --name heartdiseaseacr --sku Basic
docker tag heart-disease-api:latest heartdiseaseacr.azurecr.io/heart-disease-api:latest
docker push heartdiseaseacr.azurecr.io/heart-disease-api:latest

# Deploy
./k8s/deploy.sh deploy-all
```

---

## 📸 Assignment Screenshots

Capture these for submission:

```bash
# 1. Resources overview
kubectl get all -l app=heart-disease-api

# 2. Pod details
kubectl get pods -o wide

# 3. Service information
kubectl get svc heart-disease-api-service

# 4. Ingress details
kubectl get ingress

# 5. HPA status
kubectl get hpa

# 6. API health check
curl http://localhost:8000/health | jq

# 7. API prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_input.json | jq

# 8. Swagger UI
open http://localhost:8000/docs
# Screenshot the interactive interface

# 9. Verification script output
python3 k8s/verify_deployment.py

# 10. Helm status (if using Helm)
helm list
helm status heart-disease-api
```

---

## 🧹 Cleanup

```bash
# Using kubectl
./k8s/deploy.sh cleanup

# Using Helm
./helm/deploy-helm.sh uninstall

# Delete Minikube cluster
minikube delete

# Delete cloud clusters
# GKE:
gcloud container clusters delete heart-disease-cluster --zone=us-central1-a

# EKS:
eksctl delete cluster --name heart-disease-cluster

# AKS:
az aks delete --resource-group heart-disease-rg --name heart-disease-cluster
az group delete --name heart-disease-rg
```

---

## 📝 Quick Reference

```bash
# Deploy
./k8s/deploy.sh deploy-all

# Status
kubectl get all -l app=heart-disease-api

# Logs
kubectl logs -f -l app=heart-disease-api

# Port-forward
kubectl port-forward service/heart-disease-api-service 8000:80

# Test
curl http://localhost:8000/health
python3 k8s/verify_deployment.py

# Cleanup
./k8s/deploy.sh cleanup
```

---

## ✅ Deployment Checklist

- [ ] Docker image built and loaded
- [ ] Kubernetes cluster running
- [ ] Deployment created (3 replicas)
- [ ] Service exposed (LoadBalancer/NodePort)
- [ ] Ingress configured (optional)
- [ ] HPA enabled (optional)
- [ ] API accessible and responding
- [ ] Health endpoint working
- [ ] Prediction endpoint working
- [ ] Screenshots captured

---

**For full project documentation, see README.md**

**Last Updated:** January 2026
