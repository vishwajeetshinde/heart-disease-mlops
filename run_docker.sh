#!/bin/bash

# Heart Disease API - Docker Management Script
# Simplifies building, running, and testing the Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="heart-disease-api"
CONTAINER_NAME="heart-disease-api"
PORT=8000

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN} $1${NC}"
}

print_error() {
    echo -e "${RED} $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed"
}

check_models() {
    if [ ! -f "models/random_forest.pkl" ] || [ ! -f "models/scaler.pkl" ]; then
        print_error "Model files not found in models/ directory"
        print_info "Please train the models first by running:"
        echo "  python notebooks/feature_engineering_modeling_mlflow.py"
        exit 1
    fi
    print_success "Model files found"
}

build_image() {
    print_header "Building Docker Image"
    
    check_docker
    check_models
    
    echo "Building image: ${IMAGE_NAME}:latest"
    docker build -t ${IMAGE_NAME}:latest . || {
        print_error "Failed to build Docker image"
        exit 1
    }
    
    print_success "Docker image built successfully"
    
    # Show image info
    echo ""
    docker images ${IMAGE_NAME}:latest
}

run_container() {
    print_header "Running Docker Container"
    
    check_docker
    
    # Stop and remove existing container
    if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
        print_info "Stopping existing container..."
        docker stop ${CONTAINER_NAME} 2>/dev/null || true
        docker rm ${CONTAINER_NAME} 2>/dev/null || true
    fi
    
    echo "Starting container on port ${PORT}..."
    docker run -d \
        -p ${PORT}:8000 \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME}:latest || {
        print_error "Failed to start container"
        exit 1
    }
    
    print_success "Container started successfully"
    print_info "API available at: http://localhost:${PORT}"
    print_info "Swagger UI: http://localhost:${PORT}/docs"
    
    # Wait for container to be ready
    echo ""
    echo "Waiting for API to be ready..."
    sleep 3
    
    # Check health
    if curl -s http://localhost:${PORT}/health > /dev/null; then
        print_success "API is healthy and ready"
    else
        print_error "API health check failed"
        print_info "Check logs with: docker logs ${CONTAINER_NAME}"
    fi
}

stop_container() {
    print_header "Stopping Docker Container"
    
    if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        docker stop ${CONTAINER_NAME}
        print_success "Container stopped"
    else
        print_info "Container is not running"
    fi
}

remove_container() {
    print_header "Removing Docker Container"
    
    stop_container
    
    if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
        docker rm ${CONTAINER_NAME}
        print_success "Container removed"
    else
        print_info "Container does not exist"
    fi
}

view_logs() {
    print_header "Container Logs"
    
    if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        docker logs -f ${CONTAINER_NAME}
    else
        print_error "Container is not running"
        exit 1
    fi
}

test_api() {
    print_header "Testing API"
    
    if [ ! "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        print_error "Container is not running. Start it first with: ./run_docker.sh start"
        exit 1
    fi
    
    print_info "Running test script..."
    echo ""
    
    python test_api.py || {
        print_error "Tests failed"
        exit 1
    }
    
    print_success "Tests completed"
}

shell_access() {
    print_header "Container Shell Access"
    
    if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        docker exec -it ${CONTAINER_NAME} /bin/bash
    else
        print_error "Container is not running"
        exit 1
    fi
}

show_status() {
    print_header "Docker Status"
    
    echo "Container Status:"
    if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        print_success "Container is running"
        docker ps -f name=${CONTAINER_NAME}
    else
        print_info "Container is not running"
    fi
    
    echo ""
    echo "Image Status:"
    if [ "$(docker images -q ${IMAGE_NAME}:latest)" ]; then
        print_success "Image exists"
        docker images ${IMAGE_NAME}:latest
    else
        print_info "Image not built yet"
    fi
}

cleanup() {
    print_header "Cleanup"
    
    remove_container
    
    if [ "$(docker images -q ${IMAGE_NAME}:latest)" ]; then
        echo "Removing image..."
        docker rmi ${IMAGE_NAME}:latest
        print_success "Image removed"
    fi
    
    # Remove dangling images
    echo "Removing dangling images..."
    docker image prune -f
    
    print_success "Cleanup completed"
}

show_usage() {
    echo "Heart Disease API - Docker Management Script"
    echo ""
    echo "Usage: ./run_docker.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build     - Build Docker image"
    echo "  start     - Build and run container"
    echo "  stop      - Stop running container"
    echo "  restart   - Restart container"
    echo "  remove    - Remove container"
    echo "  logs      - View container logs"
    echo "  test      - Test API endpoints"
    echo "  shell     - Access container shell"
    echo "  status    - Show container and image status"
    echo "  cleanup   - Remove container and image"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_docker.sh start     # Build and start the API"
    echo "  ./run_docker.sh test      # Test the API"
    echo "  ./run_docker.sh logs      # View logs"
}

# Main script
case "$1" in
    build)
        build_image
        ;;
    start)
        build_image
        run_container
        ;;
    run)
        run_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        stop_container
        run_container
        ;;
    remove)
        remove_container
        ;;
    logs)
        view_logs
        ;;
    test)
        test_api
        ;;
    shell)
        shell_access
        ;;
    status)
        show_status
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        if [ -z "$1" ]; then
            show_usage
        else
            print_error "Unknown command: $1"
            echo ""
            show_usage
            exit 1
        fi
        ;;
esac
