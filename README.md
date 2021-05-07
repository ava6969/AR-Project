# ar robotics projects

## Files with code written or edited

* a2c_ppo_acktr/models.py  
* * modules written - Surreal/OpenAI/MLP_ATTN
* a2c_ppo_acktr/distributions.py 
* * modules written - MultiCategoricalDistribution/RobotARCategoricalDistribution
* a2c_ppo_acktr/arguments.py
* * added more arguments option [--vel --rec --fc --relu --save_name ]
* a2c_ppo_acktr/algo/test.py - whole file
* a2c_ppo_acktr/algo/main.py - added tensorboard writing capabilities
* a2c_ppo_acktr/algo/test.py - for testing model [whole code]
* a2c_ppo_acktr/algo/envs.py - whole code

## Requirements

* Python 3 (it might work with Python 2, but I didn't test it)
* [PyTorch](http://pytorch.org/)
* [Stable baselines3(https://github.com/DLR-RM/stable-baselines3)
* [mujoco-py](https://github.com/openai/mujoco-py) 
In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Other requirements
pip install -r requirements.txt
```


## Visualization/Test

In order to visualize the results use ```visualize.ipynb```.


## Training

#### A2C

```bash
python main.py --env-name "PongNoFrameskip-v4"
```

#### PPO

```bash
python main.py --env-name "PongNoFrameskip-v4" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01
```




### Result


