# 任务1：SSH无窗口识别

在主机执行：

```bash
cd ~/rc-comptition/wdh/robocon1
source .venv/bin/activate
python -u main.py --headless 2>&1 | tee /tmp/task1.log
```

如果任务1没有自己的虚拟环境，使用已安装任务1依赖的环境即可；进入目录不会自动切换Python环境。

--headless关闭识别预览和颜色标准窗口，保留相机采集、测量、稳定确认、动作指令和结果文件写入。普通SSH不需要DISPLAY、X11或Qt窗口。按Ctrl+C结束并释放相机。

正式结果按原流程在稳定观察完成后输出，并按配置保存到results/cube_trials.txt；不是每帧都有正式结果。第二个SSH会话可用tail -n 30 -F /tmp/task1.log查看输出。同一相机不能同时运行任务1和任务2。

不加--headless时仍使用原有窗口模式。
