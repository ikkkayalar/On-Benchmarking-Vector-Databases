container_name="weaviate_container_ann"
memory_limit="8g"
cpu_limit="8"
port_mapping_1="8080:8080"
port_mapping_2="50051:50051"
weaviate_image="cr.weaviate.io/semitechnologies/weaviate:1.27.1"
stats_file="weaviate_ann_sift_dot_docker_stats_$(date +"%Y%m%d_%H%M%S").log"

# Start the Weaviate container
docker run -d --name "$container_name" -p $port_mapping_1 -p $port_mapping_2 --memory="$memory_limit" --cpus="$cpu_limit" "$weaviate_image"

# Get the container ID based on its name
container_id=$(docker ps -aqf "name=$container_name")

# Set up ports and write to .env file
container_port=$(docker port "$container_id" | grep "8080/tcp" | awk '{print $NF}' | cut -d ':' -f 2)
cp .env .env.bk
echo "WEAVIATE_PORT=\"$container_port\"" >> .env

# Start the Python script
python3 weaviate_ann_sift_dot.py &
python_pid=$!

# Create a stats file with a header
echo "Saving Docker stats to $stats_file..."
echo "CPU%, MemoryUsage, Timestamp" > "$stats_file"

# Monitor CPU and memory usage while the Python script is running
while ps -p $python_pid > /dev/null; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    docker stats "$container_id" --no-stream --format "{{.CPUPerc}},{{.MemUsage}},$timestamp" >> "$stats_file"
    sleep 5
done

# Stop and remove the container after the script ends
docker rm -f "$container_id"
mv .env.bk .env

echo "Docker stats saved to $stats_file."
