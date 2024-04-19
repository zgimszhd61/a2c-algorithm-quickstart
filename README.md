# a2c-algorithm-quickstart

为了在Google Colab上保存视频文件并允许下载，我们可以使用`gym`环境的`Monitor`包装器来记录会话。然后，你可以将训练后的视频下载到本地机器以进行查看。下面是更新后的代码：

```python
# 安装所需库
!pip install stable-baselines3[extra] gym pyvirtualdisplay
!apt-get install -y xvfb python-opengl ffmpeg

# 导入必要的库
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from IPython.display import HTML
from pyvirtualdisplay import Display
from pathlib import Path

# 设置虚拟显示
display = Display(visible=0, size=(1400, 900))
display.start()

# 函数来创建和包装环境
def make_env(env_id, rank):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env, f'./video/{rank}', force=True)
        return env
    return _init

# 环境ID和训练步数
env_id = 'CartPole-v1'
n_envs = 4
train_steps = 10000

# 创建向量化监视器环境
env = DummyVecEnv([make_env(env_id, i) for i in range(n_envs)])

# 初始化A2C模型
model = A2C('MlpPolicy', env, verbose=1)

# 训练模型
model.learn(total_timesteps=train_steps)

# 关闭环境
env.close()

# 显示生成的视频
video_folder = './video'
video_files = [str(p) for p in Path(video_folder).glob("*.mp4")]

if video_files:
    video_path = video_files[0]
    video_file = video_path.split('/')[-1]
    # Copy the video file to a location that you can download
    !cp {video_path} /content/{video_file}
    
    # Provide a link to download
    print(f'Download your video from the following link:')
    print(f'<a href="/content/{video_file}">{video_file}</a>')
else:
    print("No video files found.")
```

这段代码首先设置了一个虚拟显示以便在没有图形界面的环境（如Colab）中渲染视频。接着定义了一个环境创建函数，并在创建环境时加入了`Monitor`包装器，它负责录制视频。模型训练结束后，代码将视频文件从存储目录复制到Colab的可下载路径，并提供了一个链接来下载视频文件。

这个方法可以让你在Colab中进行强化学习实验，并且能够查看代理在环境中的表现。
