#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HELM_RELEASE="heart-disease-api"
HELM_CHART="./helm/heart-disease-api"
NAMESPACE="default"

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
    
    # Check Helm
    if ! command -v helm &> /dev/null; then
        print_error "Helm is not installed"
        print_info "Install from: https://helm.sh/docs/intro/install/"
        exit 1
    fi
    print_success "Helm is installed ($(helm version --short))"
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi
    print_success "kubectl is installed"
    
    # Check Kubernetes
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Kubernetes cluster is not accessible"
        exit 1
    fi
    print_success "Kubernetes cluster is accessible"
    
    echo ""
}

lint_chart() {
    print_header "Linting Helm Chart"
    
    helm lint ${HELM_CHART}
    
    if [ $? -eq 0 ]; then
        print_success "Helm chart is valid"
    else
        print_error "Helm chart has errors"
        exit 1
    fi
    
    echo ""
}

dry_run() {
    print_header "Helm Dry Run"
    
    helm install ${HELM_RELEASE} ${HELM_CHART} \
        --namespace ${NAMESPACE} \
        --dry-run --debug
    
    echo ""
}

install_chart() {
    print_header "Installing Helm Chart"
    
    helm install ${HELM_RELEASE} ${HELM_CHART} \
        --namespace ${NAMESPACE} \
        --create-namespace \
        --wait \
        --timeout 5m
    
    if [ $? -eq 0 ]; then
        print_success "Helm chart installed successfully"
    else
        print_error "Helm chart installation failed"
        exit 1
    fi
    
    echo ""
}

upgrade_chart() {
    print_header "Upgrading Helm Chart"
    
    helm upgrade ${HELM_RELEASE} ${HELM_CHART} \
        --namespace ${NAMESPACE} \
        --wait \
        --timeout 5m
    
    if [ $? -eq 0 ]; then
        print_success "Helm chart upgraded successfully"
    else
        print_error "Helm chart upgrade failed"
        exit 1
    fi
    
    echo ""
}

uninstall_chart() {
    print_header "Uninstalling Helm Chart"
    
    helm uninstall ${HELM_RELEASE} --namespace ${NAMESPACE}
    
    if [ $? -eq 0 ]; then
        print_success "Helm chart uninstalled successfully"
    else
        print_error "Helm chart uninstall failed"
        exit 1
    fi
    
    echo ""
}

show_status() {
    print_header "Helm Release Status"
    
    helm status ${HELM_RELEASE} --namespace ${NAMESPACE}
    
    echo ""
    
    print_header "Kubernetes Resources"
    
    echo "Deployments:"
    kubectl get deployments -n ${NAMESPACE} -l app.kubernetes.io/instance=${HELM_RELEASE}
    echo ""
    
    echo "Pods:"
    kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/instance=${HELM_RELEASE}
    echo ""
    
    echo "Services:"
    kubectl get services -n ${NAMESPACE} -l app.kubernetes.io/instance=${HELM_RELEASE}
    echo ""
    
    echo "Ingress:"
    kubectl get ingress -n ${NAMESPACE} -l app.kubernetes.io/instance=${HELM_RELEASE}
    echo ""
}

show_values() {
    print_header "Helm Chart Values"
    
    helm get values ${HELM_RELEASE} --namespace ${NAMESPACE}
    
    echo ""
}

get_service_url() {
    print_header "Getting Service URL"
    
    SERVICE_NAME=$(kubectl get service -n ${NAMESPACE} -l app.kubernetes.io/instance=${HELM_RELEASE} -o jsonpath='{.items[0].metadata.name}')
    SERVICE_TYPE=$(kubectl get service ${SERVICE_NAME} -n ${NAMESPACE} -o jsonpath='{.spec.type}')
    
    if [ "$SERVICE_TYPE" == "LoadBalancer" ]; then
        if command -v minikube &> /dev/null && minikube status &> /dev/null; then
            print_info "Run: minikube service ${SERVICE_NAME} -n ${NAMESPACE}"
        else
            EXTERNAL_IP=$(kubectl get service ${SERVICE_NAME} -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
            if [ -z "$EXTERNAL_IP" ]; then
                print_info "LoadBalancer IP pending. Use port-forward:"
                echo -e "${YELLOW}kubectl port-forward -n ${NAMESPACE} service/${SERVICE_NAME} 8000:80${NC}"
            else
                print_success "API URL: http://${EXTERNAL_IP}"
            fi
        fi
    elif [ "$SERVICE_TYPE" == "NodePort" ]; then
        NODE_PORT=$(kubectl get service ${SERVICE_NAME} -n ${NAMESPACE} -o jsonpath='{.spec.ports[0].nodePort}')
        if command -v minikube &> /dev/null && minikube status &> /dev/null; then
            MINIKUBE_IP=$(minikube ip)
            print_success "API URL: http://${MINIKUBE_IP}:${NODE_PORT}"
        else
            print_success "API URL: http://localhost:${NODE_PORT}"
        fi
    fi
    
    echo ""
}

test_api() {
    print_header "Testing API"
    
    print_info "Starting port-forward..."
    SERVICE_NAME=$(kubectl get service -n ${NAMESPACE} -l app.kubernetes.io/instance=${HELM_RELEASE} -o jsonpath='{.items[0].metadata.name}')
    kubectl port-forward -n ${NAMESPACE} service/${SERVICE_NAME} 8000:80 &
    PF_PID=$!
    sleep 3
    
    # Test health endpoint
    print_info "Testing health endpoint..."
    HEALTH=$(curl -s http://localhost:8000/health)
    
    if [ $? -eq 0 ]; then
        print_success "Health check passed"
        echo "$HEALTH" | python3 -m json.tool
    else
        print_error "Health check failed"
    fi
    
    # Test prediction endpoint
    if [ -f "sample_input.json" ]; then
        print_info "Testing prediction endpoint..."
        PREDICTION=$(curl -s -X POST http://localhost:8000/predict \
            -H "Content-Type: application/json" \
            -d @sample_input.json)
        
        if [ $? -eq 0 ]; then
            print_success "Prediction test passed"
            echo "$PREDICTION" | python3 -m json.tool
        else
            print_error "Prediction test failed"
        fi
    fi
    
    kill $PF_PID 2>/dev/null
    echo ""
}

show_logs() {
    print_header "Showing Logs"
    
    POD=$(kubectl get pods -n ${NAMESPACE} -l app.kubernetes.io/instance=${HELM_RELEASE} -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "$POD" ]; then
        print_error "No pods found"
        exit 1
    fi
    
    print_info "Logs from pod: $POD"
    kubectl logs $POD -n ${NAMESPACE} --tail=50
}

rollback() {
    print_header "Rolling Back Release"
    
    REVISION=${2:-0}  # Default to previous revision
    
    helm rollback ${HELM_RELEASE} ${REVISION} --namespace ${NAMESPACE}
    
    if [ $? -eq 0 ]; then
        print_success "Rollback successful"
    else
        print_error "Rollback failed"
        exit 1
    fi
    
    echo ""
}

show_history() {
    print_header "Release History"
    
    helm history ${HELM_RELEASE} --namespace ${NAMESPACE}
    
    echo ""
}

# Main script
case "$1" in
    prereq)
        check_prerequisites
        ;;
    lint)
        lint_chart
        ;;
    dry-run)
        check_prerequisites
        lint_chart
        dry_run
        ;;
    install)
        check_prerequisites
        lint_chart
        install_chart
        show_status
        get_service_url
        ;;
    upgrade)
        check_prerequisites
        lint_chart
        upgrade_chart
        show_status
        ;;
    uninstall)
        uninstall_chart
        ;;
    status)
        show_status
        ;;
    values)
        show_values
        ;;
    url)
        get_service_url
        ;;
    test)
        test_api
        ;;
    logs)
        show_logs
        ;;
    rollback)
        rollback "$@"
        ;;
    history)
        show_history
        ;;
    *)
        echo "Usage: $0 {prereq|lint|dry-run|install|upgrade|uninstall|status|values|url|test|logs|rollback|history}"
        echo ""
        echo "Commands:"
        echo "  prereq      - Check prerequisites"
        echo "  lint        - Lint Helm chart"
        echo "  dry-run     - Simulate installation"
        echo "  install     - Install Helm chart"
        echo "  upgrade     - Upgrade existing release"
        echo "  uninstall   - Uninstall release"
        echo "  status      - Show release status"
        echo "  values      - Show current values"
        echo "  url         - Get service URL"
        echo "  test        - Test API endpoints"
        echo "  logs        - Show application logs"
        echo "  rollback    - Rollback to previous revision"
        echo "  history     - Show release history"
        echo ""
        echo "Examples:"
        echo "  $0 install      # Install Helm chart"
        echo "  $0 upgrade      # Upgrade release"
        echo "  $0 status       # Check status"
        echo "  $0 test         # Test API"
        exit 1
        ;;
esac
