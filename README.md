# 使用Subjects200K训练ControlAR

---

注意：这是根据ControlAR官方仓库中train_t2i_control.py修改而来，为了简单起见，去掉了混合精度、ema和DDP，这只是一个简单的示例，可以根据需要修改。

## Getting Started

---
### Installation
```bash
conda create -n ControlAR python=3.10
git clone https://github.com/Pan-Zihao/ControlAR-Subjects200K.git
cd ControlAR
pip install torch==2.1.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Pretrained Checkpoints for ControlAR
从官方项目抄过来，实际训练只需要vq、t5和llamagen（从头训不需要）
```bash
|---checkpoints
      |---t2i
            |---canny/canny_MR.safetensors
            |---hed/hed.safetensors
            |---depth/depth_MR.safetensors
            |---seg/seg_cocostuff.safetensors
            |---edge_base.safetensors
            |---depth_base.safetensors
      |---t5-ckpt
            |---flan-t5-xl
                  |---config.json
                  |---pytorch_model-00001-of-00002.bin
                  |---pytorch_model-00002-of-00002.bin
                  |---pytorch_model.bin.index.json
                  |---tokenizer.json
      |---vq
            |---vq_ds16_c2i.pt
            |---vq_ds16_t2i.pt
      |---llamagen (Only necessary for training)
            |---c2i_B_256.pt
            |---c2i_L_256.pt
            |---t2i_XL_stage2_512.pt
```
然后models文件夹里还需要dinov2_adapter的ckpt，文件夹里有说明。
### 准备数据集
由于Subjects200K在huggingface上组织的比较逆天，用起来非常不适应，所以写了一个脚本subjects200K.py重新组织了数据集。
思路是这样：

1. 将所有图片存到本地，即文件夹Subject200K_images
2. 用一个json文件来存储所有的metafile信息，数据结构是一个dict of dict，因为Subjects200K是一个和Subject有关的数据集，所以我们对于每一个Subject（比如一个cat）维护它为主体的若干张图片作为一个集合，然后每个集合也是一个字典，是image_path->caption的集合。

```bash
自定义数据集的结构长这样:
{
  subject1:{
    path1:caption1,
    path2:caption2,
    ... ...
  },
  subject2:{
    path1:caption1,
    path2:caption2,
    ... ...
  },
  ... ...
}
```

3. Subjects200K一行的那一张图片是两个528size的图片拼起来，所以需要对半裁成两个图，然后匹配上对应的caption。
```bash
python subjects200K.py
```
注意：修改成自己的路径，最后的metafile会保存成一个json文件，后面只需要用到这个metafile
## 开始训练

---
### 对图像VQ编码
使用checkpoints中下好的vq对数据集中图片进行编码，保存在code文件夹中，格式为.npy。
autoregressive/train/extract_codes_t2i.py
```bash
python autoregressive/train/extract_codes_t2i.py\
  --json-path "" --code-path "" --num 5 --vq-ckpt "" --num-workers 24
```
json-path是metafile的路径，code-path是code文件夹路径，num是需要编码的subject个数，vq-ckpt是VQ模型权重位置。

### 对caption使用t5编码
使用checkpoints中下载好的flan-t5-xl对图片对应的caption进行编码，保存在caption_embds文件夹中，格式为.npy。
language/extract_t5_features.py
```bash
python language/extract_t5_features.py\
  --json-path "" --t5-path "" --num "" --t5-model-path "" --code-dir ""
```
code-dir是code文件夹路径，t5-path和t5-model-path都是t5的权重路径。

### 训练
文件train_t2i_subject.py，参数太多了，对着default看一下，主要是一些路径，其他默认的都不需要改，应该也挺简单的。

## 数据集设计及训练思路
ControlAR的官方实现是输入两个condition，一个是condition image，另一个是text condition，text condition是已经编码好的t5 embeddings，存在文件里。condition image是根据condition type选择处理方式的，我这里参考的lineart，因为它输入的也是一个图片。然后condition的image输入进来是需要处理的，比如lineart是用了一个现成的网络去提取一张图片的lineart，这是一个单通道的tensor，这里为了匹配模型输入的维度，我就直接对三个通道取平均了（均匀池化）。

所以一个ControlAR的input和ouput是这样：
```bash
ControlAR(code, condition_image, tex_embedding, attn_mask)->code
```
现在想要在Subjects200K上训练Customization Generation能力，也就是以一个Subject的图片为condition，然后根据caption生成其他同一个Subject的图片。这时候就需要一个数据集，当Dataloader取出一个图片时，我需要得到与它相同Subject的图片训练。所以设计了一个上述的Dataset，get_item的逻辑是从一个image_list中随机取出一个图片，然后在构建的时候会顺带生成一个反向的map，它从图片路径映射到Subject。

所以取数据的流程如下：
1. 随机从images_list中取出一张图片；
2. 根据这个图片map到它对应的Subject、code_path、caption_embds_path；
3. 根据Subject从json中找到这个图像集合，从这个set中随机抽出一个图片作为condition_image（去掉了取出的那一张图片，以免重复）。这样所有的图片都可能作condition。
4. 后面根据官方代码训练即可。

## 讨论
这个方法很简单啊，但是ControlAR是一个对应token控制，用condition_image的每一个token控制对应位置的token预测，这种控制方式不知道能不能提取出subject的语义然后学习，我感觉是不行...