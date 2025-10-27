FROM python:3.9-slim

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app/ ./app/
COPY ml/ ./ml/

# 创建非root用户
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "app/main.py"]