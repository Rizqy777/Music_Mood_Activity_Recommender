from __future__ import annotations

import argparse
import base64
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import _load_env_files

PROJECT_TAG = "music-mood-activity-web"


EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".vscode",
    ".specstory",
    "__pycache__",
    "chats",
    "logs",
    "notebooks",
    "tools",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".odt", ".pdf", ".ipynb"}


def should_include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    parts = set(relative.parts)
    if parts & EXCLUDED_DIRS:
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if relative.parts and relative.parts[0] == "datasets":
        return False
    if relative.parts and relative.parts[0] == "data_lake":
        return len(relative.parts) >= 2 and relative.parts[1] == "recommender"
    return True


def make_bundle() -> Path:
    target = Path(tempfile.gettempdir()) / f"{PROJECT_TAG}-{int(time.time())}.zip"
    
    # 1. Primero creamos y empaquetamos el ZIP
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in ROOT.rglob("*"):
            if file_path.is_file() and should_include(file_path):
                zf.write(file_path, file_path.relative_to(ROOT))
                
    # 2. Ahora que el archivo ya existe y está cerrado, medimos su peso
    size_mb = target.stat().st_size / (1024 * 1024)
    print(f"Tamaño del empaquetado a subir: {size_mb:.2f} MB")
    
    return target


def get_clients(region: str):
    return (
        boto3.client("ec2", region_name=region),
        boto3.client("s3", region_name=region),
        boto3.client("ssm", region_name=region),
    )


def ensure_bucket(s3, bucket: str, region: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
        return
    except ClientError as exc:
        error_code = str(exc.response.get("Error", {}).get("Code", ""))
        if error_code in {"403", "AccessDenied"}:
            return
        if error_code not in {"404", "NoSuchBucket", "NotFound"}:
            raise
    kwargs = {"Bucket": bucket}
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**kwargs)
    s3.get_waiter("bucket_exists").wait(Bucket=bucket)


def latest_amazon_linux_ami(ssm) -> str:
    response = ssm.get_parameter(
        Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
    )
    return response["Parameter"]["Value"]


def default_vpc_id(ec2) -> str:
    response = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])
    vpcs = response.get("Vpcs", [])
    if not vpcs:
        raise RuntimeError("No se encontro una VPC por defecto en esta cuenta AWS.")
    return vpcs[0]["VpcId"]


def ensure_security_group(ec2, vpc_id: str) -> str:
    name = f"{PROJECT_TAG}-sg"
    response = ec2.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [name]},
            {"Name": "vpc-id", "Values": [vpc_id]},
        ]
    )
    groups = response.get("SecurityGroups", [])
    if groups:
        group_id = groups[0]["GroupId"]
    else:
        created = ec2.create_security_group(
            GroupName=name,
            Description="HTTP access for Music Mood Activity web app",
            VpcId=vpc_id,
            TagSpecifications=[
                {
                    "ResourceType": "security-group",
                    "Tags": [{"Key": "Project", "Value": PROJECT_TAG}],
                }
            ],
        )
        group_id = created["GroupId"]
    for port in (80,):
        try:
            ec2.authorize_security_group_ingress(
                GroupId=group_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": port,
                        "ToPort": port,
                        "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                    }
                ],
            )
        except ClientError as exc:
            if "InvalidPermission.Duplicate" not in str(exc):
                raise
    return group_id


def terminate_existing(ec2) -> None:
    response = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Project", "Values": [PROJECT_TAG]},
            {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
        ]
    )
    instance_ids = [
        instance["InstanceId"]
        for reservation in response.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    if not instance_ids:
        return
    ec2.terminate_instances(InstanceIds=instance_ids)
    ec2.get_waiter("instance_terminated").wait(InstanceIds=instance_ids)
def build_user_data(bundle_url: str) -> str:
    script = f"""#!/bin/bash
set -euxo pipefail
dnf update -y
dnf install -y python3.11 python3.11-pip unzip shadow-utils

# --- NUEVO: CREAR MEMORIA RAM VIRTUAL (SWAP) DE 4GB ---
# Esto evita que el OOM Killer colapse la web al cargar los modelos
dd if=/dev/zero of=/swapfile bs=1M count=4096
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
# -----------------------------------------------------

python3.11 -m ensurepip --upgrade || true
id -u appuser >/dev/null 2>&1 || useradd --system --create-home --shell /sbin/nologin appuser

# 1. Creamos la carpeta principal en el disco duro real
mkdir -p /opt/music-mood

# 2. Descargamos el ZIP directamente en /opt/ (saltándonos la RAM)
python3.11 - <<'PY'
import urllib.request
urllib.request.urlretrieve("{bundle_url}", "/opt/music-mood.zip")
PY

# 3. Limpiamos y descomprimimos usando el archivo que ahora vive en /opt
rm -rf /opt/music-mood/app
mkdir -p /opt/music-mood/app
unzip -q /opt/music-mood.zip -d /opt/music-mood/app

# Seguimos con la instalación normal de tu entorno
cd /opt/music-mood/app
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip

if [ -f requirements-web.txt ]; then
  .venv/bin/pip install -r requirements-web.txt
else
  .venv/bin/pip install -r requirements.txt
fi

.venv/bin/python - <<'PY'
import web_app
print("PREIMPORT_OK", web_app.app.title)
PY

chown -R appuser:appuser /opt/music-mood

cat >/etc/systemd/system/music-mood-web.service <<'SERVICE'
[Unit]
Description=Music Mood Activity Web
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/opt/music-mood/app
Environment=PYTHONUNBUFFERED=1
AmbientCapabilities=CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
ExecStart=/opt/music-mood/app/.venv/bin/python -m uvicorn web_app:app --host 0.0.0.0 --port 80
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable --now music-mood-web
sleep 8
systemctl status music-mood-web --no-pager -l || true
journalctl -u music-mood-web -n 80 --no-pager || true
"""
    return base64.b64encode(script.encode("utf-8")).decode("ascii")

def deploy(args: argparse.Namespace) -> None:
    _load_env_files()
    region = args.region or os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION") or "us-east-1"
    bucket = args.bucket or os.getenv("AWS_BUCKET_NAME")
    if not bucket:
        raise RuntimeError("Define AWS_BUCKET_NAME en .env o pasa --bucket.")

    ec2, s3, ssm = get_clients(region)
    bundle = make_bundle()
    key = f"deployments/webapp/{bundle.name}"
    ensure_bucket(s3, bucket, region)
    s3.upload_file(str(bundle), bucket, key)
    bundle_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=3600,
    )

    if not args.keep_existing:
        terminate_existing(ec2)

    vpc_id = default_vpc_id(ec2)
    security_group_id = ensure_security_group(ec2, vpc_id)
    ami_id = args.ami_id or latest_amazon_linux_ami(ssm)
    user_data = build_user_data(bundle_url)

    response = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=args.instance_type,
        MinCount=1,
        MaxCount=1,
        SecurityGroupIds=[security_group_id],
        UserData=user_data,
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/xvda",  # Volumen raíz por defecto para Amazon Linux
                "Ebs": {
                    "VolumeSize": 30,       # Tamaño en GB
                    "VolumeType": "gp3",    # gp3 es la generación actual: más rápida y económica
                    "DeleteOnTermination": True
                }
            }
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": PROJECT_TAG},
                    {"Key": "Project", "Value": PROJECT_TAG},
                ],
            }
        ],
    )
    instance_id = response["Instances"][0]["InstanceId"]
    ec2.get_waiter("instance_running").wait(InstanceIds=[instance_id])
    details = ec2.describe_instances(InstanceIds=[instance_id])
    instance = details["Reservations"][0]["Instances"][0]
    public_dns = instance.get("PublicDnsName")
    public_ip = instance.get("PublicIpAddress")
    url = f"http://{public_dns or public_ip}"

    print("Despliegue completado.")
    print(f"Instancia: {instance_id}")
    print(f"URL: {url}")
    print("La primera carga puede tardar unos minutos mientras systemd instala dependencias y arranca Uvicorn.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Despliega la web unificada en AWS EC2.")
    parser.add_argument("--region", default=None, help="Region AWS. Por defecto usa AWS_DEFAULT_REGION o us-east-1.")
    parser.add_argument("--bucket", default=None, help="Bucket S3 para subir el paquete de despliegue.")
    parser.add_argument("--instance-type", default=os.getenv("AWS_WEB_INSTANCE_TYPE", "t3.medium"))
    parser.add_argument("--ami-id", default=None, help="AMI opcional. Por defecto usa Amazon Linux 2023.")
    parser.add_argument("--keep-existing", action="store_true", help="No termina instancias anteriores del mismo proyecto.")
    return parser.parse_args()


if __name__ == "__main__":
    deploy(parse_args())
