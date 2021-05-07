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

## Training

#### A2C

```bash


python main.py --algo a2c  --num-processes 10 --log-interval 100 --recurrent-policy --use-linear-lr-decay --num-env-steps 100000000 --num-steps 25 --use-gae --save_name pos --rec 64 --fc '64 64 64' --lr 5e-4 --entropy-coef 0 --value-loss-coef 1 --seed 13

```


### Enjoy

```bash
python test.py 
```

