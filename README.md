# pytorch_peot_rnn
基于pytorch_rnn的古诗词生成

# 说明
config.py里面含有训练、测试、预测的参数，更改后运行：
```python
main.py
```

# 预测结果
```python
if config.do_predict:
	result = trainer.generate('丽日照残春')
	print("".join(result))
	result = trainer.gen_acrostic('深度学习')
	print("".join(result))
	
丽日照残春，
风光摇落时。
不知花发意，
不得见春风。

深山高下有余灵，万里无人见钓矶。
度日茱萸人不得，一枝不得不相见。
学舞一枝花落叶，不知何处是君王。
习书不见金闺后，应是君王赐手间。
```

# 参考
> https://github.com/chenyuntc/pytorch-book<br>
其中第九章的古诗词生成，修改了以下地方：<br>
1、重构了代码架构；<br>
2、增加了数据集生成的过程；<br>
3、RNN网络改为batch_first；<br>
4、计算损失时不计算padding部分；<br>