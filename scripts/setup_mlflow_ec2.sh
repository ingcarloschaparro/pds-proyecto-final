#!/bin/bash
# Script para configurar MLflow en AWS EC2
# Uso: ./setup_mlflow_ec2.sh

echo "🚀 Configurando MLflow en AWS EC2..."

# Actualizar sistema
echo "📦 Actualizando sistema..."
sudo yum update -y

# Instalar dependencias
echo "🐍 Instalando Python y dependencias..."
sudo yum install -y python3 python3-pip git

# Instalar MLflow y dependencias
echo "🔬 Instalando MLflow..."
pip3 install mlflow boto3 psycopg2-binary

# Crear estructura de directorios
echo "📁 Creando estructura de directorios..."
mkdir -p ~/mlflow-server/{experiments,models,artifacts,scripts,logs}

# Crear script de inicio
echo "📝 Creando script de inicio..."
cat > ~/mlflow-server/start_mlflow.sh << 'EOF'
#!/bin/bash
cd ~/mlflow-server
nohup mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 4 \
    > logs/mlflow.log 2>&1 &

echo "MLflow server iniciado en puerto 5000"
echo "PID: $!"
echo $! > mlflow.pid
EOF

chmod +x ~/mlflow-server/start_mlflow.sh

# Crear script de parada
echo "🛑 Creando script de parada..."
cat > ~/mlflow-server/stop_mlflow.sh << 'EOF'
#!/bin/bash
if [ -f mlflow.pid ]; then
    PID=$(cat mlflow.pid)
    kill $PID
    rm mlflow.pid
    echo "MLflow server detenido"
else
    echo "No se encontró PID del servidor MLflow"
fi
EOF

chmod +x ~/mlflow-server/stop_mlflow.sh

echo "✅ Configuración completada!"
echo "📍 Para iniciar MLflow: ~/mlflow-server/start_mlflow.sh"
echo "📍 Para detener MLflow: ~/mlflow-server/stop_mlflow.sh"
echo "📍 Logs: ~/mlflow-server/logs/mlflow.log"
