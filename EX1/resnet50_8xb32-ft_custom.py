_base_ = [
    '../_base_/models/resnet50.py',           # 模型设置
    '../_base_/datasets/imagenet_bs32.py',    # 数据设置
    '../_base_/schedules/imagenet_bs256.py',  # 训练策略设置
    '../_base_/default_runtime.py',           # 运行设置
]

# 模型设置
model = dict(
    backbone=dict(
        frozen_stages=2,
        init_cfg=dict(
            type='Pretrained',
            checkpoint = "F:\\py\\ECE371_Assigment\\t1\\resnet50_8xb32_in1k_20210831-ea4938fc.pth",
            prefix='backbone',
        )),
    head=dict(
        num_classes=5,  # 你的类别数量
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
)

# 数据设置
data_root = 'data/custom_dataset'
train_dataloader = dict(
    batch_size=256,
    dataset=dict(
        type='CustomDataset',  # 使用 CustomDataset
        data_root=data_root,
        ann_file='train.txt',  # 指定训练集标注文件
        data_prefix='',
        classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],  # 明确指定类别名称
    ))
val_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='val.txt',  # 指定验证集标注文件
        data_prefix='',
        classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],
    ))
test_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        ann_file='val.txt',  # 如果没有单独的测试集，可以使用验证集
        data_prefix='val',
        classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],
    ))

# 训练策略设置
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),  # 或 type='Adam'·
)   # 优化器设置学习率（原始训练可能用 0.1，Fine-tuning 建议 0.001~0.0001）
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1
    # 0.1表示学习率降至10%，by_epoch=True（默认）若设为False，则按迭代次数（iteration）触发（需配合total_iters参数）
)

# 指定工作目录
work_dir = 'F:/py/ECE371_Assigment/t1/work_dir'
# ————————————————————————————————————————————————————————————————————————————————————————————————
#以下为个人笔记
#Epoch（训练轮次）
# 定义：整个训练数据集被模型完整遍历一次的次数，同一数据集反复训练多次（如100个Epochs）
# 小数据集可能需要更多Epochs（如50~100），大数据集可能只需少量（如10~20）。
# Batch size（批次大小）：每次训练所使用的样本数，通常为16、32、64、128、256等。
# 作用：决定了模型的收敛速度、精度。大批量通常需要更少迭代次数，但每个迭代耗时更长。
# 学习率（Learning rate）模型更新的速度，决定了模型的收敛速度、精度。
# 优化器（Optimizer）
# 定义：模型更新的算法，如SGD、Adam、Adagrad等。
# 作用：模型更新的算法决定了模型的收敛速度、精度。
# 学习率衰减（Learning rate decay），学习率随着训练的进行而衰减，防止模型过拟合。
# 权重衰减（Weight decay）模型参数更新时，对权重的惩罚，防止过拟合。
# Dropout率
# 定义：在前向传播时随机丢弃神经元的概率（如0.3表示30%神经元被置零）。
# 作用：防止过拟合，提高模型的泛化能力。
# ————————————————————————————————————————————————————————————————————————————————————————————————
# 这些参数的一般取值范围：
# 学习率：0.01~0.1
# Batch size：16~128
# 优化器：SGD、Adam、Adagrad等
# 学习率衰减：0.1~0.001
# 权重衰减：0.0001~0.001
# Dropout率：0.1~0.5
# Momentum：0.9~0.99，用于优化器（如SGD with Momentum）的参数，模拟物理中的动量效应，加速梯度下降并减少震荡。
# 作用：加速收敛：在梯度方向一致时累积速度。逃离局部极小值：通过惯性越过平坦区域。
# ————————————————————————————————————————————————————————————————————————————————————————————————
