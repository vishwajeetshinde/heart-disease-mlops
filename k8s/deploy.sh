#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="heart-disease-api"
IMAGE_TAG="latest"
NAMESPACE="default"
DEPLOYMENT_NAME="heart-disease-api"
SERVICE_NAME="heart-disease-api-service"

# Functions
print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_success "Docker is installed"
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi
    print_success "kubectl is installed"
    
    # Check if Kubernetes is running
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Kubernetes cluster is not accessible"
        print_info "Please start Docker Desktop Kubernetes or Minikube"
        exit 1
    fi
    print_success "Kubernetes cluster is accessible"
    
    echo ""
}

build_docker_image() {
    print_header "Building Docker Image"
    
    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile not found in current directory"
        exit 1
    fi
    
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
    
    echo ""
}

load_image_to_k8s() {
    print_header "Loading Image to Kubernetes"
    
    # Check if using Minikube
    if command -v minikube &> /dev/null && minikube status &> /dev/null; then
        print_info "Loading image to Minikube..."
        minikube image load ${IMAGE_NAME}:${IMAGE_TAG}
        print_success "Image loaded to Minikube"
    else
        print_info "Using Docker Desktop Kubernetes - image already available"
    fi
    
    echo ""
}

deploy_with_kubectl() {
    print_header "Deploying with kubectl"
    
    # Apply ConfigMap
    if [ -f "k8s/configmap.yaml" ]; then
        kubectl apply -f k8s/configmap.yaml
        print_success "ConfigMap applied"
    fi
    
    # Apply Deployment
    kubectl apply -f k8s/deployment.yaml
    print_success "Deployment applied"
    
    # Wait for deployment
    print_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/${DEPLOYMENT_NAME} --timeout=300s
    
    if [ $? -eq 0 ]; then
        print_success "Deployment is ready"
    else
        print_error "Deployment failed"
        kubectl get pods
        exit 1
    fi
    
    echo ""
}

deploy_service() {
    print_header "Deploying Service"
    
    # Check if using LoadBalancer or NodePort
    if [[ "$1" == "nodeport" ]]; then
        kubectl apply -f k8s/service-nodeport.yaml
        print_success "NodePort service deployed"
    else
        kubectl apply -f k8s/deployment.yaml  # Contains LoadBalancer service
        print_success "LoadBalancer service deployed"
    fi
    
    echo ""
}

deploy_ingress() {
    print_header "Deploying Ingress"
    
    # Check if Ingress controller is installed
    if ! kubectl get ingressclass nginx &> /dev/null; then
        print_info "Installing NGINX Ingress Controller..."
        
        if command -v minikube &> /dev/null && minikube status &> /dev/null; then
            minikube addons enable ingress
        else
            kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
        fi
        
        print_info "Waiting for Ingress controller to be ready..."
        sleep 10
    fi
    
    # Apply Ingress
    kubectl apply -f k8s/ingress.yaml
    print_success "Ingress deployed"
    
    # Add to /etc/hosts
    print_info "Add this line to /etc/hosts:"
    if command -v minikube &> /dev/null && minikube status &> /dev/null; then
        echo -e "${YELLOW}$(minikube ip) heart-disease-api.local${NC}"
    else
        echo -e "${YELLOW}127.0.0.1 heart-disease-api.local${NC}"
    fi
    
    echo ""
}

deploy_hpa() {
    print_header "Deploying Horizontal Pod Autoscaler"
    
    # Check if metrics-server is installed
    if ! kubectl get deployment metrics-server -n kube-system &> /dev/null; then
        print_info "Metrics server not found. HPA requires metrics-server."
        print_info "To install: kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml"
    else
        kubectl apply -f k8s/hpa.yaml
        print_success "HPA deployed"
    fi
    
    echo ""
}

get_service_url() {
    print_header "Getting Service URL"
    
    SERVICE_TYPE=$(kubectl get service ${SERVICE_NAME} -o jsonpath='{.spec.type}' 2>/dev/null)
    
    if [ "$SERVICE_TYPE" == "LoadBalancer" ]; then
        print_info "Waiting for LoadBalancer IP..."
        
        if command -v minikube &> /dev/null && minikube status &> /dev/null; then
            print_info "Run: minikube service ${SERVICE_NAME}"
            minikube service ${SERVICE_NAME} --url
        else
            kubectl get service ${SERVICE_NAME}
            EXTERNAL_IP=$(kubectl get service ${SERVICE_NAME} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
            
            if [ -z "$EXTERNAL_IP" ]; then
                print_info "LoadBalancer IP pending. Using port-forward instead:"
                echo -e "${YELLOW}kubectl port-forward service/${SERVICE_NAME} 8000:80${NC}"
            else
                print_success "API URL: http://${EXTERNAL_IP}"
            fi
        fi
    elif [ "$SERVICE_TYPE" == "NodePort" ]; then
        NODE_PORT=$(kubectl get service heart-disease-api-nodeport -o jsonpath='{.spec.ports[0].nodePort}')
        
        if command -v minikube &> /dev/null && minikube status &> /dev/null; then
            MINIKUBE_IP=$(minikube ip)
            print_success "API URL: http://${MINIKUBE_IP}:${NODE_PORT}"
        else
            print_success "API URL: http://localhost:${NODE_PORT}"
        fi
    fi
    
    echo ""
}

verify_deployment() {
    print_header "Verifying Deployment"
    
    # Get pods
    echo "Pods:"
    kubectl get pods -l app=${DEPLOYMENT_NAME}
    echo ""
    
    # Get services
    echo "Services:"
    kubectl get services
    echo ""
    
    # Get ingress
    echo "Ingress:"
    kubectl get ingress
    echo ""
    
    # Test API using port-forward
    print_info "Testing API with port-forward..."
    kubectl port-forward service/${SERVICE_NAME} 8000:80 &
    PF_PID=$!
    sleep 3
    
    HEALTH_CHECK=$(curl -s http://localhost:8000/health 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        print_success "Health check passed"
        echo "$HEALTH_CHECK"
    else
        print_error "Health check failed"
    fi
    
    kill $PF_PID 2>/dev/null
    echo ""
}

show_status() {
    print_header "Deployment Status"
    
    echo "Deployments:"
    kubectl get deployments
    echo ""
    
    echo "Pods:"
    kubectl get pods
    echo ""
    
    echo "Services:"
    kubectl get services
    echo ""
    
    echo "Ingress:"
    kubectl get ingress
    echo ""
    
    if kubectl get hpa ${DEPLOYMENT_NAME} &> /dev/null; then
        echo "HPA:"
        kubectl get hpa
        echo ""
    fi
}

cleanup() {
    print_header "Cleaning Up Resources"
    
    kubectl delete -f k8s/deployment.yaml --ignore-not-found=true
    kubectl delete -f k8s/service-nodeport.yaml --ignore-not-found=true
    kubectl delete -f k8s/ingress.yaml --ignore-not-found=true
    kubectl delete -f k8s/configmap.yaml --ignore-not-found=true
    kubectl delete -f k8s/hpa.yaml --ignore-not-found=true
    
    print_success "Resources cleaned up"
    echo ""
}

show_logs() {
    print_header "Showing Logs"
    
    POD=$(kubectl get pods -l app=${DEPLOYMENT_NAME} -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$POD" ]; then
        print_error "No pods found"
        exit 1
    fi
    
    print_info "Logs from pod: $POD"
    kubectl logs $POD --tail=50
}

# Main script
case "$1" in
    prereq)
        check_prerequisites
        ;;
    build)
        check_prerequisites
        build_docker_image
        load_image_to_k8s
        ;;
    deploy)
        check_prerequisites
        deploy_with_kubectl
        deploy_service "$2"
        get_service_url
        ;;
    deploy-all)
        check_prerequisites
        build_docker_image
        load_image_to_k8s
        deploy_with_kubectl
        deploy_service "$2"
        deploy_ingress
        deploy_hpa
        get_service_url
        verify_deployment
        ;;
    ingress)
        deploy_ingress
        ;;
    hpa)
        deploy_hpa
        ;;
    verify)
        verify_deployment
        ;;
    status)
        show_status
        ;;
    url)
        get_service_url
        ;;
    logs)
        show_logs
        ;;
    cleanup)
        cleanup
        ;;
    *)
        echo "Usage: $0 {prereq|build|deploy|deploy-all|ingress|hpa|verify|status|url|logs|cleanup}"
        echo ""
        echo "Commands:"
        echo "  prereq      - Check prerequisites"
        echo "  build       - Build and load Docker image"
        echo "  deploy      - Deploy to Kubernetes (add 'nodeport' for NodePort service)"
        echo "  deploy-all  - Full deployment (build + deploy + ingress + hpa)"
        echo "  ingress     - Deploy Ingress"
        echo "  hpa         - Deploy Horizontal Pod Autoscaler"
        echo "  verify      - Verify deployment and test API"
        echo "  status      - Show deployment status"
        echo "  url         - Get service URL"
        echo "  logs        - Show application logs"
        echo "  cleanup     - Delete all resources"
        echo ""
        echo "Examples:"
        echo "  $0 deploy-all           # Full deployment with LoadBalancer"
        echo "  $0 deploy-all nodeport  # Full deployment with NodePort"
        echo "  $0 build                # Build Docker image only"
        echo "  $0 deploy               # Deploy with kubectl"
        echo "  $0 status               # Check deployment status"
        exit 1
        ;;
esac
