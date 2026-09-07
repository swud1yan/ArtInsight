# ArtInsight 阿里云部署说明

这套部署方式保留现有 PySide6 桌面程序，通过 noVNC 在浏览器里访问完整 GUI。它适合毕业设计验收：功能不用重写，老师打开链接即可操作。

## 1. 上线数据范围

只上传两个 50 风格数据集：

```text
data/demo_1000_dataset
data/expanded_50_style_dataset
```

不要上传：

```text
data/raw
data/course_dataset
data/paper_dataset
data/datasets/wikiart_four_styles
data/cache
data/exports
data/models/*.backup_*.joblib
.venv
.git
.idea
```

## 2. 服务器目录

推荐放在：

```bash
/opt/artinsight
```

最终结构：

```text
/opt/artinsight/
  app/
  config/
  controllers/
  core/
  data/
    demo_1000_dataset/
    expanded_50_style_dataset/
    models/
    cache/
    exports/
  deploy/
  docker-compose.artinsight.yml
  main.py
  requirements.txt
```

## 3. 上传代码和数据

代码可以用 Git、WinSCP 或压缩包上传。数据建议用 WinSCP 或 rsync，断点续传更稳。

Git Bash 或 WSL 示例：

```bash
rsync -avh --progress --partial /d/PCA/data/demo_1000_dataset/ root@服务器IP:/opt/artinsight/data/demo_1000_dataset/
rsync -avh --progress --partial /d/PCA/data/expanded_50_style_dataset/ root@服务器IP:/opt/artinsight/data/expanded_50_style_dataset/
```

## 4. 可选：准备预置数据库

如果希望老师直接从“提取特征/训练模型”开始，而不是重新导入数据，可以在本地先生成老师版数据库：

```powershell
python scripts\prepare_teacher_database.py --server-root /opt/artinsight
```

生成：

```text
data/artinsight_teacher.sqlite3
```

上传后改名为：

```bash
mv /opt/artinsight/data/artinsight_teacher.sqlite3 /opt/artinsight/data/artinsight.sqlite3
```

如果希望老师完整体验“导入数据集”，可以不上传数据库，让程序自动创建空数据库。

## 5. 启动容器

服务器安装 Docker 和 Docker Compose 后，在 `/opt/artinsight` 执行：

```bash
cp deploy/artinsight.env.example .env
nano .env
docker compose -f docker-compose.artinsight.yml --env-file .env up -d --build
```

查看日志：

```bash
docker logs -f artinsight-novnc
```

本机验证：

```bash
curl http://127.0.0.1:6080/
```

浏览器地址：

```text
http://服务器IP:6080/vnc.html?autoconnect=true&resize=scale
```

如果设置了 `ARTINSIGHT_VNC_PASSWORD`，页面会要求输入密码。

## 6. Nginx 反向代理

已有后端不用动，ArtInsight 只占本机 `127.0.0.1:6080`。新增一个子域名更稳，例如：

```nginx
server {
    listen 80;
    server_name art.example.com;

    location / {
        proxy_pass http://127.0.0.1:6080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600;
    }
}
```

重载 Nginx：

```bash
nginx -t
systemctl reload nginx
```

阿里云安全组只需要放行 80/443。如果临时不用域名，也可以放行 6080，但正式给老师建议走 Nginx。

## 7. 老师演示流程

推荐先用小数据集跑完整流程：

```text
数据集管理 -> 导入 data/demo_1000_dataset
模型训练 -> 提取特征 -> 开始训练模型
图片上传 -> 分类/相似检索
PCA 分析 -> 风格地图 -> 报告导出
```

如果要体验完整 50 风格大数据集：

```text
数据集管理 -> 导入 data/expanded_50_style_dataset
模型训练 -> 继续提取缺失特征 -> 开始训练模型
```

`expanded_50_style_dataset` 有两万多张图，特征提取会比较久，建议提前跑一次。
