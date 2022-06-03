# Texas Cheat'Em
<img src="https://github.com/Raingate/Texas-Cheat-Em/blob/master/p1196583759.jpg" width = "320" height = "200" div align=center />

This is a course project of SJTU AI3617. 
Members include: Wenzhuo Zheng, Mingyang Jiang, Jinghao Feng

Limited Texas Hold'em is a world-wide popular card game, and also a typical imperfect information multi-agent game. It's a compound of fortune and strategy. This project studies application of MARL algorithms on this game. We first study performance of existing RL algorithms, and propose some customized modifications based on experimental observations. Then, we integrate our ideas to propose a brand new value-aware and rival-aware MARL algorithm that outperforms all baselines.

# Running
- Check dependencies in `requirements.txt`.


- If you want to reproduce our results in our paper, run `train.py`.

Example:
```shell
python train.py --player1 cheater --player2 random
```  



- If you want to load pretrained model and evaluate, run `evaluate.py`.

Example:
```shell
python evaluate.py --player1 cheater --player2 random --load_path1 path_to_your_model
```

- If you want to train the cheating sheet, run `train_table.py`.

Example:
```shell
python train_table.py
```

Default output directory is `networks`.

- If you want to play with your model, run `Human_vs_AI.py`.

Example:
```shell
python Human_vs_AI.py --agent cheater --load_path path_to_your_model
```

Have fun!
