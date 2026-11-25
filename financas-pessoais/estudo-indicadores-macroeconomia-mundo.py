# Title: estudo-indicadores-macroeconomia-mundo

# Set environment variables for sagemaker_studio imports

import os
os.environ['DataZoneProjectId'] = 'attutj3b25vbtz'
os.environ['DataZoneDomainId'] = 'dzd-bemsmvoo5tmvon'
os.environ['DataZoneEnvironmentId'] = '5f8477vlhqwysn'
os.environ['DataZoneDomainRegion'] = 'us-east-1'

# create both a function and variable for metadata access
_resource_metadata = None

def _get_resource_metadata():
    global _resource_metadata
    if _resource_metadata is None:
        _resource_metadata = {
            "AdditionalMetadata": {
                "DataZoneProjectId": "attutj3b25vbtz",
                "DataZoneDomainId": "dzd-bemsmvoo5tmvon",
                "DataZoneEnvironmentId": "5f8477vlhqwysn",
                "DataZoneDomainRegion": "us-east-1",
            }
        }
    return _resource_metadata
metadata = _get_resource_metadata()

"""
Logging Configuration

Purpose:
--------
This sets up the logging framework for code executed in the user namespace.
"""

from typing import Optional


def _set_logging(log_dir: str, log_file: str, log_name: Optional[str] = None):
    import os
    import logging
    from logging.handlers import RotatingFileHandler

    level = logging.INFO
    max_bytes = 5 * 1024 * 1024
    backup_count = 5

    # fallback to /tmp dir on access, helpful for local dev setup
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        log_dir = "/tmp/kernels/"

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger() if not log_name else logging.getLogger(log_name)
    logger.handlers = []
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Rotating file handler
    fh = RotatingFileHandler(filename=log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Logging initialized for {log_name}.")


_set_logging("/var/log/computeEnvironments/kernel/", "kernel.log")
_set_logging("/var/log/studio/data-notebook-kernel-server/", "metrics.log", "metrics")

import logging
from sagemaker_studio import ClientConfig, sqlutils, sparkutils, dataframeutils

logger = logging.getLogger(__name__)
logger.info("Initializing sparkutils")
spark = sparkutils.init()
logger.info("Finished initializing sparkutils")

def _reset_os_path():
    """
    Reset the process's working directory to handle mount timing issues.
    
    This function resolves a race condition where the Python process starts
    before the filesystem mount is complete, causing the process to reference
    old mount paths and inodes. By explicitly changing to the mounted directory
    (/home/sagemaker-user), we ensure the process uses the correct, up-to-date
    mount point.
    
    The function logs stat information (device ID and inode) before and after
    the directory change to verify that the working directory is properly
    updated to reference the new mount.
    
    Note:
        This is executed at module import time to ensure the fix is applied
        as early as possible in the kernel initialization process.
    """
    try:
        import os
        import logging

        logger = logging.getLogger(__name__)
        logger.info("---------Before------")
        logger.info("CWD: %s", os.getcwd())
        logger.info("stat('.'): %s %s", os.stat('.').st_dev, os.stat('.').st_ino)
        logger.info("stat('/home/sagemaker-user'): %s %s", os.stat('/home/sagemaker-user').st_dev, os.stat('/home/sagemaker-user').st_ino)

        os.chdir("/home/sagemaker-user")

        logger.info("---------After------")
        logger.info("CWD: %s", os.getcwd())
        logger.info("stat('.'): %s %s", os.stat('.').st_dev, os.stat('.').st_ino)
        logger.info("stat('/home/sagemaker-user'): %s %s", os.stat('/home/sagemaker-user').st_dev, os.stat('/home/sagemaker-user').st_ino)
    except Exception as e:
        logger.exception(f"Failed to reset working directory: {e}")

_reset_os_path()

# MARKDOWN CELL 3n9o
# Texto em linguagem markdown.

import pandas as pd

# Carrega o arquivo carregado na mesma pasta
df_indicadores_macro = pd.read_excel('indices_macro.xlsx')

df_indicadores_macro.head()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar a planilha (que você já subiu com sucesso!)
df_indicadores_macro = pd.read_excel('indices_macro.xlsx')

# 2. Filtrar os dados a partir de 1990
df_filtrado = df_indicadores_macro[df_indicadores_macro['ano'] >= 1990]

# 3. Agrupar os dados por região e por ano, somando o PIB Nominal
df_agrupado = df_filtrado.groupby(['paises_regioes', 'ano'])['pib_nominal_usd'].sum().reset_index()

# 4. Converter o PIB de dólares brutos para "Trilhões de USD" para melhor leitura
df_agrupado['pib_nominal_trilhoes'] = df_agrupado['pib_nominal_usd'] / 1e12

# 5. Configurar o estilo visual do gráfico
plt.figure(figsize=(13, 7))
sns.set_theme(style="whitegrid")

# Criar o gráfico de linhas
sns.lineplot(
    data=df_agrupado, 
    x='ano', 
    y='pib_nominal_trilhoes', 
    hue='paises_regioes', 
    marker='o', 
    linewidth=2
)

# Customizar títulos e legendas
plt.title('Evolução do PIB Nominal por Região (Desde 1990)', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Ano', fontsize=12)
plt.ylabel('PIB Nominal (Trilhões de USD)', fontsize=12)
plt.xlim(1990, 2025)

# Posicionar legenda para fora do gráfico para não poluir o visual
plt.legend(title='Regiões', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

plt.tight_layout()

# Exibir o gráfico na tela
plt.show()

from IPython.display import display, HTML

# Lê o conteúdo do HTML gerado e exibe diretamente na saída do notebook
with open('Artigo_Cientifico_PIB.html', 'r', encoding='utf-8') as f:
    conteudo_html = f.read()

display(HTML(conteudo_html))

!jupyter nbconvert --to html --no-input ind_macro_202611.ipynb --output relatorio_sem_codigo

import boto3
from botocore.config import Config

arquivo = 'relatorio_sem_codigo.html'
bucket = 'amazon-sagemaker-533266997012-us-east-1-e434f9f6d26d'
chave = f'dzd-bemsmvoo5tmvon/attutj3b25vbtz/dev/shared/{arquivo}'

s3 = boto3.client(
    's3',
    region_name='us-east-1',
    endpoint_url='https://s3.us-east-1.amazonaws.com',
    config=Config(signature_version='s3v4'),
)

url = s3.generate_presigned_url(
    'get_object',
    Params={'Bucket': bucket, 'Key': chave,
            'ResponseContentDisposition': f'attachment; filename="{arquivo}"'},
    ExpiresIn=3600,
)
print(url)

import boto3

bucket = 'amazon-sagemaker-533266997012-us-east-1-e434f9f6d26d'
chave = 'dzd-bemsmvoo5tmvon/attutj3b25vbtz/dev/shared/relatorio_sem_codigo.html'

boto3.client('s3').delete_object(Bucket=bucket, Key=chave)
print('Arquivo removido do S3')

import boto3

bucket = 'amazon-sagemaker-533266997012-us-east-1-e434f9f6d26d'
prefixo = 'dzd-bemsmvoo5tmvon/attutj3b25vbtz/dev/'

s3 = boto3.client('s3')
total = 0
for pagina in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=prefixo):
    for obj in pagina.get('Contents', []):
        total += obj['Size']
        print(f"{obj['Size']/1024:>10.1f} KB  {obj['LastModified']:%Y-%m-%d %H:%M}  {obj['Key']}")

print(f'\nTotal: {total/1024/1024:.2f} MB')