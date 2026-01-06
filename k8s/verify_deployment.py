#!/usr/bin/env python3
"""
Kubernetes Deployment Verification Script
Tests deployed API endpoints and generates deployment screenshots
"""

import requests
import json
import time
import subprocess
import sys
from datetime import datetime

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.YELLOW}ℹ️  {text}{Colors.END}")

def run_command(cmd, capture_output=True):
    """Run shell command and return output"""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode == 0, result.stdout.strip()
        else:
            result = subprocess.run(cmd, shell=True)
            return result.returncode == 0, ""
    except Exception as e:
        return False, str(e)

def check_kubernetes_status():
    """Check Kubernetes cluster and deployment status"""
    print_header("Checking Kubernetes Status")
    
    # Check cluster info
    success, output = run_command("kubectl cluster-info")
    if success:
        print_success("Kubernetes cluster is accessible")
    else:
        print_error("Cannot access Kubernetes cluster")
        return False
    
    # Check deployments
    success, output = run_command("kubectl get deployments -l app=heart-disease-api -o json")
    if success:
        deployments = json.loads(output)
        if deployments['items']:
            for dep in deployments['items']:
                name = dep['metadata']['name']
                replicas = dep['status'].get('replicas', 0)
                ready = dep['status'].get('readyReplicas', 0)
                print_success(f"Deployment: {name} - {ready}/{replicas} replicas ready")
        else:
            print_error("No deployments found")
            return False
    
    # Check pods
    success, output = run_command("kubectl get pods -l app=heart-disease-api -o json")
    if success:
        pods = json.loads(output)
        if pods['items']:
            for pod in pods['items']:
                name = pod['metadata']['name']
                phase = pod['status']['phase']
                if phase == 'Running':
                    print_success(f"Pod: {name} - Status: {phase}")
                else:
                    print_info(f"Pod: {name} - Status: {phase}")
        else:
            print_error("No pods found")
            return False
    
    # Check services
    success, output = run_command("kubectl get services -l app=heart-disease-api -o json")
    if success:
        services = json.loads(output)
        if services['items']:
            for svc in services['items']:
                name = svc['metadata']['name']
                svc_type = svc['spec']['type']
                print_success(f"Service: {name} - Type: {svc_type}")
        else:
            print_error("No services found")
            return False
    
    return True

def get_service_endpoint():
    """Get the API endpoint URL"""
    print_header("Getting Service Endpoint")
    
    # Check service type
    success, output = run_command("kubectl get service heart-disease-api-service -o jsonpath='{.spec.type}'")
    if not success:
        # Try nodeport service
        success, output = run_command("kubectl get service heart-disease-api-nodeport -o jsonpath='{.spec.type}'")
        service_name = "heart-disease-api-nodeport"
    else:
        service_name = "heart-disease-api-service"
    
    service_type = output
    
    if service_type == "LoadBalancer":
        # Check if using Minikube
        success, _ = run_command("which minikube")
        if success:
            success, url = run_command(f"minikube service {service_name} --url")
            if success and url:
                print_success(f"API URL (Minikube): {url}")
                return url
        
        # Check LoadBalancer IP
        success, ip = run_command(f"kubectl get service {service_name} -o jsonpath='{{.status.loadBalancer.ingress[0].ip}}'")
        if success and ip:
            url = f"http://{ip}"
            print_success(f"API URL (LoadBalancer): {url}")
            return url
        else:
            print_info("LoadBalancer IP pending, using port-forward")
            return None
    
    elif service_type == "NodePort":
        success, node_port = run_command(f"kubectl get service {service_name} -o jsonpath='{{.spec.ports[0].nodePort}}'")
        
        # Check if using Minikube
        success, _ = run_command("which minikube")
        if success:
            success, minikube_ip = run_command("minikube ip")
            if success and minikube_ip:
                url = f"http://{minikube_ip}:{node_port}"
                print_success(f"API URL (Minikube NodePort): {url}")
                return url
        
        url = f"http://localhost:{node_port}"
        print_success(f"API URL (NodePort): {url}")
        return url
    
    return None

def start_port_forward():
    """Start kubectl port-forward in background"""
    print_info("Starting port-forward to access API...")
    
    cmd = "kubectl port-forward service/heart-disease-api-service 8000:80"
    
    # Start port-forward in background
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)  # Wait for port-forward to establish
    
    return proc

def test_api_endpoints(base_url):
    """Test all API endpoints"""
    print_header("Testing API Endpoints")
    
    results = {}
    
    # Test 1: Root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print_success(f"GET / - Status: {response.status_code}")
            results['root'] = True
        else:
            print_error(f"GET / - Status: {response.status_code}")
            results['root'] = False
    except Exception as e:
        print_error(f"GET / - Error: {str(e)}")
        results['root'] = False
    
    # Test 2: Health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_success(f"GET /health - Status: {response.status_code}")
            print(f"  Model loaded: {data.get('model_loaded', False)}")
            print(f"  Scaler loaded: {data.get('scaler_loaded', False)}")
            results['health'] = True
        else:
            print_error(f"GET /health - Status: {response.status_code}")
            results['health'] = False
    except Exception as e:
        print_error(f"GET /health - Error: {str(e)}")
        results['health'] = False
    
    # Test 3: Model info endpoint
    try:
        response = requests.get(f"{base_url}/model-info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_success(f"GET /model-info - Status: {response.status_code}")
            print(f"  Model type: {data.get('model_type', 'N/A')}")
            print(f"  Features: {data.get('n_features', 'N/A')}")
            results['model_info'] = True
        else:
            print_error(f"GET /model-info - Status: {response.status_code}")
            results['model_info'] = False
    except Exception as e:
        print_error(f"GET /model-info - Error: {str(e)}")
        results['model_info'] = False
    
    # Test 4: Prediction endpoint
    sample_data = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=sample_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print_success(f"POST /predict - Status: {response.status_code}")
            print(f"  Prediction: {data.get('prediction', 'N/A')}")
            print(f"  Confidence: {data.get('confidence', 0):.2%}")
            print(f"  Risk Level: {data.get('risk_level', 'N/A')}")
            results['predict'] = True
        else:
            print_error(f"POST /predict - Status: {response.status_code}")
            results['predict'] = False
    except Exception as e:
        print_error(f"POST /predict - Error: {str(e)}")
        results['predict'] = False
    
    return results

def generate_deployment_report(results):
    """Generate deployment verification report"""
    print_header("Deployment Verification Report")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"Timestamp: {timestamp}\n")
    
    # Kubernetes Status
    print("Kubernetes Resources:")
    run_command("kubectl get all -l app=heart-disease-api", capture_output=False)
    
    print("\n" + "=" * 60)
    
    # API Test Results
    print("\nAPI Endpoint Test Results:")
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for endpoint, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {endpoint}: {status}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print_success(f"\n🎉 All tests passed! Deployment verified successfully.")
        return True
    else:
        print_error(f"\n⚠️  Some tests failed. Please check the deployment.")
        return False

def capture_kubectl_status():
    """Capture kubectl status for screenshots"""
    print_header("Capturing Deployment Status")
    
    commands = [
        ("kubectl get deployments", "Deployments"),
        ("kubectl get pods", "Pods"),
        ("kubectl get services", "Services"),
        ("kubectl get ingress", "Ingress"),
        ("kubectl get hpa", "HPA (if exists)"),
    ]
    
    for cmd, description in commands:
        print(f"\n{description}:")
        run_command(cmd, capture_output=False)

def main():
    """Main execution function"""
    print_header("Kubernetes Deployment Verification")
    
    # Check Kubernetes status
    if not check_kubernetes_status():
        print_error("Kubernetes deployment check failed")
        sys.exit(1)
    
    # Get service endpoint
    api_url = get_service_endpoint()
    
    # If no direct URL, use port-forward
    port_forward_proc = None
    if not api_url:
        port_forward_proc = start_port_forward()
        api_url = "http://localhost:8000"
    
    # Wait a bit for everything to be ready
    print_info("Waiting for services to be ready...")
    time.sleep(5)
    
    # Test API endpoints
    test_results = test_api_endpoints(api_url)
    
    # Generate report
    all_passed = generate_deployment_report(test_results)
    
    # Capture status for screenshots
    capture_kubectl_status()
    
    # Cleanup port-forward if started
    if port_forward_proc:
        port_forward_proc.terminate()
        print_info("Port-forward stopped")
    
    # Print instructions
    print_header("Next Steps")
    print("1. Take screenshots of the kubectl status output above")
    print("2. Test the API using the URLs shown")
    print("3. Access Swagger docs at: {}/docs".format(api_url))
    print("4. Monitor logs: kubectl logs -l app=heart-disease-api")
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
