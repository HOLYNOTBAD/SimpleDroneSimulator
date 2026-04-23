# Simple Drone Interception Simulator

这是一个用于课堂展示的无人机拦截仿真器。当前版本包含两个视觉伺服拦截控制器，并提供三类目标运动场景：

- ICRA26 PS-LOS 控制器
- TCST High-Speed IBVS-SO3 控制器
- 来向目标、直线逃跑目标、S 形逃跑目标

仿真默认先快速计算完整过程，结束后再打开 3D 轨迹和第一视角可视化窗口。

## 1. 环境配置

推荐使用 Conda 创建独立环境：

```powershell
conda create -y -n SimpleDroneSim python=3.11 numpy pyyaml matplotlib
conda activate SimpleDroneSim
```

进入项目目录：

```powershell
cd D:\Desktop\Interception_class\Simple-Drone-Simulator
```

可选：安装为 editable package。脚本已经内置路径处理，不安装也能运行，但安装后更规范。

```powershell
python -m pip install -e .
```

如果 matplotlib 窗口无法弹出，可以安装 Qt 后端：

```powershell
conda install -y pyqt
```

## 2. 快速运行

### ICRA26 PS-LOS 控制器

来向目标：

```powershell
python .\scripts\ibvs_so3_ctrl_sim.py --config .\configs\ibvs_so3_head_on.yaml
```

直线逃跑目标：

```powershell
python .\scripts\ibvs_so3_ctrl_sim.py --config .\configs\ibvs_so3_straight_escape.yaml
```

S 形逃跑目标：

```powershell
python .\scripts\ibvs_so3_ctrl_sim.py --config .\configs\ibvs_so3_s_escape.yaml
```

### TCST High-Speed IBVS-SO3 控制器

来向目标：

```powershell
python .\scripts\high_speed_ibvs_so3_ctrl_sim.py --config .\configs\high_speed_ibvs_so3_head_on.yaml
```

直线逃跑目标：

```powershell
python .\scripts\high_speed_ibvs_so3_ctrl_sim.py --config .\configs\high_speed_ibvs_so3_straight_escape.yaml
```

S 形逃跑目标：

```powershell
python .\scripts\high_speed_ibvs_so3_ctrl_sim.py --config .\configs\high_speed_ibvs_so3_s_escape.yaml
```

## 3. 输出说明

每次仿真结束后，终端只输出两行：

```text
hit: True
t_hit: 8.245 s
```

含义：

- `hit`：是否成功拦截。
- `t_hit`：首次进入命中半径的仿真时间。

当前默认命中半径为 `2.0 m`，配置项在每个 YAML 文件中：

```yaml
termination:
  hit_radius: 2.0
```

## 4. 三类测试场景

### 来向目标

目标从前方接近拦截器，并带有横向速度。用于展示基础拦截过程。

配置文件：

- `configs/ibvs_so3_head_on.yaml`
- `configs/high_speed_ibvs_so3_head_on.yaml`

### 直线逃跑目标

目标从前方沿远离方向直线逃跑。该场景主要考察追赶速度和纵向收敛能力。

配置文件：

- `configs/ibvs_so3_straight_escape.yaml`
- `configs/high_speed_ibvs_so3_straight_escape.yaml`

### S 形逃跑目标

目标向前逃跑，同时横向做正弦机动。该场景更能体现视线约束和横向机动能力。

配置文件：

- `configs/ibvs_so3_s_escape.yaml`
- `configs/high_speed_ibvs_so3_s_escape.yaml`

S 形运动参数：

```yaml
target:
  mode: s_escape
  s_amplitude_m: 8.0
  s_frequency_hz: 0.14
```

## 5. 控制器代码位置

ICRA26 PS-LOS 控制器：

```text
control/ibvs_so3_controller.py
```

TCST High-Speed IBVS-SO3 控制器：

```text
control/high_speed_ibvs_so3_controller.py
```

目标运动模型：

```text
models/target.py
```

主要仿真脚本：

```text
scripts/ibvs_so3_ctrl_sim.py
scripts/high_speed_ibvs_so3_ctrl_sim.py
```

## 6. 可视化说明

默认配置：

```yaml
visualization:
  realtime: false
  enable_realtime_animation: false
  enable_offline_animation: true
```

这表示：

- 仿真运行时不实时刷新窗口，因此速度较快。
- 仿真结束后才打开可视化窗口。
- 3D 窗口显示拦截器和目标轨迹。
- 第一视角窗口显示目标在相机视场中的位置。

如果只想看终端结果，不想打开窗口，可以临时使用非交互后端：

```powershell
$env:MPLBACKEND='Agg'
python .\scripts\ibvs_so3_ctrl_sim.py --config .\configs\ibvs_so3_head_on.yaml
Remove-Item Env:MPLBACKEND
```

## 7. 常用参数修改

修改命中半径：

```yaml
termination:
  hit_radius: 2.0
```

修改目标初始位置和速度：

```yaml
tgt0:
  p_e: [70.0, 0.0, 0.0]
  v_e: [-6.0, 6.0, 0.0]
```

修改 3D 显示范围：

```yaml
visualization:
  map_center: [35.0, 10.0, -5.0]
  map_size: [120.0, 90.0, 60.0]
```

修改第一视角目标显示大小：

```yaml
visualization:
  fov_target_diameter_m: 1.5
  fov_target_marker_size_max: 42.0
```

## 8. 课堂展示建议

推荐展示顺序：

1. 运行 ICRA 来向目标场景，说明基本拦截流程。
2. 运行 TCST 与 ICRA 的直线逃跑场景，对比追赶能力。
3. 运行 TCST 与 ICRA 的 S 形逃跑场景，对比机动目标下的视线保持能力。

建议使用 PowerShell 或 VS Code 终端运行命令。仿真结束后关闭 matplotlib 窗口，脚本才会完全退出。

协作说明：本项目在开发过程中接受了协作者的共同参与与改进建议。
