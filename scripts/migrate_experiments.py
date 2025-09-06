#!/usr/bin/env python3
"""
Script para migrar experimentos MLflow de local a EC2
Uso: python migrate_experiments.py --ec2-ip IP_EC2 --local-path ./mlruns
"""

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

def create_tarball(local_mlruns_path, output_path):
    """Crear tarball con experimentos MLflow"""
    print(f"📦 Creando tarball de {local_mlruns_path}...")
    
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(local_mlruns_path, arcname="mlruns")
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Tarball creado: {output_path} ({size_mb:.2f} MB)")
    return output_path

def transfer_to_ec2(tarball_path, ec2_ip, key_path, username="ec2-user"):
    """Transferir tarball a EC2 usando SCP"""
    print(f"🚀 Transferiendo a EC2 ({ec2_ip})...")
    
    remote_path = f"{username}@{ec2_ip}:/home/{username}/mlflow-experiments.tar.gz"
    cmd = ["scp", "-i", key_path, tarball_path, remote_path]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Transferencia completada")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en transferencia: {e}")
        return False

def extract_on_ec2(ec2_ip, key_path, username="ec2-user"):
    """Extraer tarball en EC2"""
    print(f"📂 Extrayendo en EC2...")
    
    commands = [
        "cd ~/mlflow-server",
        "tar -xzf ~/mlflow-experiments.tar.gz",
        "mv mlruns/* experiments/ 2>/dev/null || true",
        "rm ~/mlflow-experiments.tar.gz",
        "ls -la experiments/"
    ]
    
    cmd = ["ssh", "-i", key_path, f"{username}@{ec2_ip}", "; ".join(commands)]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Extracción completada")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en extracción: {e}")
        print(e.stderr)
        return False

def verify_mlflow_server(ec2_ip, port=5000):
    """Verificar que el servidor MLflow esté ejecutándose"""
    print(f"🔍 Verificando servidor MLflow en {ec2_ip}:{port}...")
    
    try:
        import requests
        response = requests.get(f"http://{ec2_ip}:{port}", timeout=10)
        if response.status_code == 200:
            print("✅ Servidor MLflow está ejecutándose")
            return True
        else:
            print(f"⚠️ Servidor respondió con código {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ No se pudo conectar al servidor: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Migrar experimentos MLflow a EC2")
    parser.add_argument("--ec2-ip", required=True, help="IP de la instancia EC2")
    parser.add_argument("--local-path", default="./mlruns", help="Ruta local de mlruns")
    parser.add_argument("--key-path", required=True, help="Ruta al archivo .pem")
    parser.add_argument("--username", default="ec2-user", help="Usuario de EC2")
    
    args = parser.parse_args()
    
    # Verificar que existe la ruta local
    if not os.path.exists(args.local_path):
        print(f"❌ No se encontró la ruta local: {args.local_path}")
        return 1
    
    # Crear tarball
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tarball_path = tmp.name
    
    try:
        create_tarball(args.local_path, tarball_path)
        
        # Transferir a EC2
        if not transfer_to_ec2(tarball_path, args.ec2_ip, args.key_path, args.username):
            return 1
        
        # Extraer en EC2
        if not extract_on_ec2(args.ec2_ip, args.key_path, args.username):
            return 1
        
        # Verificar servidor
        if verify_mlflow_server(args.ec2_ip):
            print("🎉 Migración completada exitosamente!")
            print(f"🌐 Accede a MLflow en: http://{args.ec2_ip}:5000")
        else:
            print("⚠️ Migración completada, pero el servidor no está ejecutándose")
            print("💡 Inicia el servidor con: ~/mlflow-server/start_mlflow.sh")
        
        return 0
        
    finally:
        # Limpiar archivo temporal
        if os.path.exists(tarball_path):
            os.unlink(tarball_path)

if __name__ == "__main__":
    exit(main())
