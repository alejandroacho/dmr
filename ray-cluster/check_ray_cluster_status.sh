export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^ray-node-(head|worker)$' | head -1)
echo "Found container: $VLLM_CONTAINER"
docker exec $VLLM_CONTAINER ray status
