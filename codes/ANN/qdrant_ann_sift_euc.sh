container_name="qdrant_container_ann"
memory_limit="8g"
cpu_limit="8"
port_mapping="6922:6333"

docker run -d --name "$container_name" -p $port_mapping --memory="$memory_limit" --cpus="$cpu_limit" qdrant/qdrant

container_id=$(docker ps -aqf "name=$container_name")

container_port=$(docker port "$container_id" | grep "tcp" | awk '{print $NF}' | cut -d ':' -f 2)
cp .env .env.bk
echo "QDRANT_PORT=\"$container_port\"" >> .env

python3.10 qdrant_ann_sift_euc.py &

python_pid=$!

while ps -p $python_pid > /dev/null; do

    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    docker stats "$container_id" --no-stream --format "{{.CPUPerc}},{{.MemUsage}},$timestamp"    
    sleep 5

done

docker rm -f "$container_id"
mv .env.bk .env