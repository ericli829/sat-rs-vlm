# 本地环境

## 新环境

```bash
python scripts/environment/bootstrap_local.py --with-dev
```

该命令创建或复用 `.venv`，安装基础和开发依赖，不安装模型栈。真实模型 smoke
再增加 `--with-model`。使用 `--clean` 会显式删除并重建指定 venv。

激活方式：

```powershell
# PowerShell
.venv\Scripts\Activate.ps1
```

```bat
rem CMD
.venv\Scripts\activate.bat
```

```bash
# Linux/macOS
source .venv/bin/activate
```

## 当前开发机约定

当前仓库本地验证使用默认环境：

```powershell
python -m pytest -q
```

当前实际解释器是系统默认的 Windows Store Python 3.11。不要使用现有 `.venv`，
因为其依赖不完整。路径变量参考
`environments/local.env.example`，不要把个人绝对路径写入 YAML。

## 能力检查

```bash
python scripts/environment/check_environment.py
python scripts/environment/export_environment.py --output reports/environment/local
```

无 GPU 对基础检查不是错误。`--require-model` 和 `--require-gpu` 只应在真实模型
任务前使用。
