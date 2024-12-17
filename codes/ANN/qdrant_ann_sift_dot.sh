container_name="qdrant_container_ann"
memory_limit="8g"
cpu_limit="8"
port_mapping="6922:6333"
stats_file="qdrant_ann_sift_dot_docker_stats_$(date +"%Y%m%d_%H%M%S").log"

# Start Qdrant container
docker run -d --name "$container_name" -p $port_mapping --memory="$memory_limit" --cpus="$cpu_limit" qdrant/qdrant

container_id=$(docker ps -aqf "name=$container_name")

# Get container port and update environment file
container_port=$(docker port "$container_id" | grep "tcp" | awk '{print $NF}' | cut -d ':' -f 2)
cp .env .env.bk
echo "QDRANT_PORT=\"$container_port\"" >> .env

# Start Python script
python3 qdrant_ann_gist_cos.py &
python_pid=$!

# Save Docker stats to a file
echo "Saving Docker stats to $stats_file..."
echo "CPU%, MemoryUsage, Timestamp" > "$stats_file"  # Add header to the stats file

while ps -p $python_pid > /dev/null; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    docker stats "$container_id" --no-stream --format "{{.CPUPerc}},{{.MemUsage}},$timestamp" >> "$stats_file"
    sleep 5
done

# Clean up after script execution
docker rm -f "$container_id"
mv .env.bk .env
echo "Docker stats saved to $stats_file."
