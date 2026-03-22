# Quickstart

This guide assumes you are using Minikube and the `porto` namespace.

## Optional: Build Images
**Note:** In the code, they are automatically pulled from my dockerhub.

```bash
docker build -t mlops-porto-training -f docker/Dockerfile.training .
docker build -t mlops-porto-inference -f docker/Dockerfile.inference .
docker build -t mlops-porto-streamlit -f docker/Dockerfile.streamlit .
```

## Basic Minikube Management Commands

```bash
minikube start --driver docker
minikube stop
minikube status
minikube delete
```

## Minikube Prereqs (Storage Addons)

These are required for StatefulSets/PVCs on Minikube:

```bash
minikube addons enable storage-provisioner
minikube addons enable default-storageclass
```

## Apply Manifests (Order Matters)

```bash
# First apply namespace and switch to it
minikube kubectl -- apply -f k8s/00-namespace.yml
minikube kubectl -- config set-context --current --namespace=porto

# Then we can apply the following
minikube kubectl -- apply -f k8s/01-secrets.yml
minikube kubectl -- apply -f k8s/02-configmaps.yml
minikube kubectl -- apply -f k8s/03-storage.yml
minikube kubectl -- apply -f k8s/05-seed-jobs.yml
```

Wait until data migration is completed...

```bash
minikube kubectl -- apply -f k8s/10-infra.yml
minikube kubectl -- apply -f k8s/15-data-prep.yml
```

Wait until data prep is completed...

```bash
minikube kubectl -- apply -f k8s/20-training.yml
```

Wait until training jobs are completed...

```bash
minikube kubectl -- apply -f k8s/25-model-selector.yml
minikube kubectl -- apply -f k8s/30-inference.yml
```

## Access Services

List services:

```bash
minikube kubectl -- -n porto get svc
```

List all service URLs:

```bash
minikube service -n porto --all
```

Get specific URLs:

```bash
minikube service -n porto mlflow-server --url
minikube service -n porto grafana --url
minikube service -n porto streamlit --url
```

Port-forward (most reliable for ClusterIP services):

```bash
# This way we can create a tunnel to the service and inspect in browser eg. http://localhost:port
minikube kubectl -- -n porto port-forward svc/mlflow-server 5000:5000
minikube kubectl -- -n porto port-forward svc/grafana 3000:3000
minikube kubectl -- -n porto port-forward svc/streamlit 8501:8501
```

## Quick Health Checks
**Note:** Here I check Streamlit, but it is also important to make sure that other services like Grafana and MLflow are operating well.

Pods status:

```bash
minikube kubectl -- -n porto get pods -o wide
```

Logs (tail):

```bash
minikube kubectl -- -n porto logs -l app=streamlit --tail=100
```

Logs (follow):

```bash
minikube kubectl -- -n porto logs -l app=streamlit -f
```

Events:

```bash
minikube kubectl -- -n porto get events --sort-by=.lastTimestamp | tail -n 30
```

Restarts / OOMs:

```bash
minikube kubectl -- -n porto get pods --sort-by=.status.containerStatuses[0].restartCount
```

Describe a pod:

```bash
minikube kubectl -- -n porto describe pod -l app=streamlit
```

PVC status:

```bash
# Get all pvc resources
minikube kubectl -- get pvc

# Get total capacity of pvcs
minikube kubectl -- get pvc | grep -i "bound" | awk '{print $4}' | awk -F 'M' '{sum+=$1}END{print sum}'
```

## Optional: Detailed Log Inspection
**Note:** There is also a health check script `scripts/healthcheck.sh`, that runs the key checks/logs in one go. Use it like:

```bash
bash ./scripts/healthcheck.sh -n porto -o healthcheck.log
```

## Optional: Cleanup
**Note:** There is a TTL controller set in code, that already deals with cleanup, but this could be done manually as follows.

Delete training jobs:

```bash
minikube kubectl -- -n porto delete job -l app=training
```

Remove completed pods:

```bash
minikube kubectl -- -n porto delete pod --field-selector=status.phase=Succeeded
```

Delete all deployments quickly:

```bash
minikube kubectl -- -n porto delete $(minikube kubectl -- get all | grep "^deploy" | awk '{print $1}')
```

## Minikube Resource Usage

Inside the Minikube node:

```bash
minikube ssh -- free -h
minikube ssh -- df -h
minikube ssh -- top -b -n 1 | head -n 20
```

Kubernetes metrics (requires metrics-server):

```bash
minikube addons enable metrics-server
minikube kubectl -- top nodes
minikube kubectl -- top pods -A
```

If using Docker driver:

```bash
docker stats
```

Minikube profile info:

```bash
minikube profile list -d
```
