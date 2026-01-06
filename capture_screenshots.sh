#!/bin/bash

# Screenshot Capture Guide for Kubernetes Deployment
# Heart Disease MLOps - Task 7

echo "================================================================"
echo "SCREENSHOT CAPTURE GUIDE - Kubernetes Deployment"
echo "================================================================"
echo ""
echo "Follow these steps and take screenshots of each output:"
echo ""

# Screenshot 1
echo "📸 SCREENSHOT 1: Kubernetes Resources Overview"
echo "----------------------------------------------------------------"
echo "Run this command and take a screenshot:"
echo ""
echo "  kubectl get all -l app=heart-disease-api"
echo ""
read -p "Press Enter after taking screenshot 1..."
kubectl get all -l app=heart-disease-api
echo ""
echo ""

# Screenshot 2
echo "📸 SCREENSHOT 2: Pod Details"
echo "----------------------------------------------------------------"
echo "Run this command and take a screenshot:"
echo ""
echo "  kubectl get pods -o wide"
echo ""
read -p "Press Enter after taking screenshot 2..."
kubectl get pods -o wide
echo ""
echo ""

# Screenshot 3
echo "📸 SCREENSHOT 3: Service Details"
echo "----------------------------------------------------------------"
echo "Run this command and take a screenshot:"
echo ""
echo "  kubectl get service heart-disease-api-service"
echo ""
read -p "Press Enter after taking screenshot 3..."
kubectl get service heart-disease-api-service
echo ""
echo ""

# Screenshot 4
echo "📸 SCREENSHOT 4: Deployment Details"
echo "----------------------------------------------------------------"
echo "Run this command and take a screenshot:"
echo ""
echo "  kubectl describe deployment heart-disease-api"
echo ""
read -p "Press Enter after taking screenshot 4..."
kubectl describe deployment heart-disease-api | head -50
echo ""
echo ""

# Screenshot 5
echo "📸 SCREENSHOT 5: Ingress Configuration"
echo "----------------------------------------------------------------"
echo "Run this command and take a screenshot:"
echo ""
echo "  kubectl get ingress"
echo ""
read -p "Press Enter after taking screenshot 5..."
kubectl get ingress
echo ""
echo ""

# Screenshot 6
echo "📸 SCREENSHOT 6: HPA (Horizontal Pod Autoscaler)"
echo "----------------------------------------------------------------"
echo "Run this command and take a screenshot:"
echo ""
echo "  kubectl get hpa"
echo ""
read -p "Press Enter after taking screenshot 6..."
kubectl get hpa
echo ""
echo ""

# Screenshot 7
echo "📸 SCREENSHOT 7: API Health Check"
echo "----------------------------------------------------------------"
echo "Starting port-forward in background..."
kubectl port-forward service/heart-disease-api-service 8000:80 > /dev/null 2>&1 &
PF_PID=$!
sleep 3

echo "Run this command and take a screenshot:"
echo ""
echo "  curl http://localhost:8000/health | jq"
echo ""
read -p "Press Enter after taking screenshot 7..."
curl -s http://localhost:8000/health | python3 -m json.tool
echo ""
echo ""

# Screenshot 8
echo "📸 SCREENSHOT 8: API Model Info"
echo "----------------------------------------------------------------"
echo "Run this command and take a screenshot:"
echo ""
echo "  curl http://localhost:8000/model-info | jq"
echo ""
read -p "Press Enter after taking screenshot 8..."
curl -s http://localhost:8000/model-info | python3 -m json.tool
echo ""
echo ""

# Screenshot 9
echo "📸 SCREENSHOT 9: API Prediction Request"
echo "----------------------------------------------------------------"
echo "Run this command and take a screenshot:"
echo ""
echo "  curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d @sample_input.json | jq"
echo ""
read -p "Press Enter after taking screenshot 9..."
if [ -f "sample_input.json" ]; then
    curl -s -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d @sample_input.json | python3 -m json.tool
else
    echo "⚠️  sample_input.json not found in current directory"
fi
echo ""
echo ""

# Screenshot 10
echo "📸 SCREENSHOT 10: Swagger UI (Interactive API Docs)"
echo "----------------------------------------------------------------"
echo "Open your browser and go to:"
echo ""
echo "  http://localhost:8000/docs"
echo ""
echo "Take a screenshot of the Swagger UI page showing all endpoints."
echo ""
read -p "Press Enter to open browser..."
if command -v open &> /dev/null; then
    open http://localhost:8000/docs
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8000/docs
else
    echo "Please manually open: http://localhost:8000/docs"
fi
echo ""
read -p "Press Enter after taking screenshot 10..."
echo ""

# Screenshot 11
echo "📸 SCREENSHOT 11: Pod Logs"
echo "----------------------------------------------------------------"
echo "Run this command and take a screenshot:"
echo ""
POD_NAME=$(kubectl get pods -l app=heart-disease-api -o jsonpath='{.items[0].metadata.name}')
echo "  kubectl logs $POD_NAME --tail=30"
echo ""
read -p "Press Enter after taking screenshot 11..."
if [ ! -z "$POD_NAME" ]; then
    kubectl logs $POD_NAME --tail=30
else
    echo "⚠️  No pods found"
fi
echo ""
echo ""

# Screenshot 12 (Optional - if using Helm)
echo "📸 SCREENSHOT 12 (Optional): Helm Release"
echo "----------------------------------------------------------------"
echo "If you deployed with Helm, run:"
echo ""
echo "  helm list"
echo "  helm status heart-disease-api"
echo ""
read -p "Press Enter to check Helm..."
if command -v helm &> /dev/null; then
    helm list
else
    echo "Helm not installed or not used - skip this screenshot"
fi
echo ""
echo ""

# Cleanup
kill $PF_PID 2>/dev/null

echo "================================================================"
echo "✅ SCREENSHOT CAPTURE COMPLETE!"
echo "================================================================"
echo ""
echo "You should now have 11-12 screenshots covering:"
echo "  1. Kubernetes resources overview"
echo "  2. Pod details and status"
echo "  3. Service configuration"
echo "  4. Deployment details"
echo "  5. Ingress configuration"
echo "  6. HPA (autoscaling)"
echo "  7. API health check"
echo "  8. API model info"
echo "  9. API prediction response"
echo "  10. Swagger UI (browser)"
echo "  11. Pod logs"
echo "  12. Helm status (if used)"
echo ""
echo "📁 Save all screenshots in a folder for your assignment submission."
echo ""
