container_name="weaviate_container_ann"
memory_limit="8g"
cpu_limit="8"
port_mapping_1="8080:8080"
port_mapping_2="50051:50051"
weaviate_image="cr.weaviate.io/semitechnologies/weaviate:1.27.1"

# Weaviate container’ını başlat
docker run -d --name "$container_name" -p $port_mapping_1 -p $port_mapping_2 --memory="$memory_limit" --cpus="$cpu_limit" "$weaviate_image"

# Container ID’yi container ismine göre yakala
container_id=$(docker ps -aqf "name=$container_name")

# Portları ayarla ve .env dosyasına yaz
container_port=$(docker port "$container_id" | grep "8080/tcp" | awk '{print $NF}' | cut -d ':' -f 2)
cp .env .env.bk
echo "WEAVIATE_PORT=\"$container_port\"" >> .env

# Weaviate için Python script’ini çalıştır
python3.10 weaviate_ann_gist_dot.py &

# Python script’inin PID’ini yakala
python_pid=$!

# Python script’i çalışırken CPU ve bellek kullanımını izle
while ps -p $python_pid > /dev/null; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    docker stats "$container_id" --no-stream --format "{{.CPUPerc}},{{.MemUsage}},$timestamp"
    sleep 5
done

# Script bitince container’ı durdur ve sil
docker rm -f "$container_id"
mv .env.bk .env