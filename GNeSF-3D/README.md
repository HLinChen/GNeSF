

## How to Use

### Download pretrained model
The method needs to be trained for 3D surface reconstruction following NeuralRecon. Please download the pretrained model from the [link](https://drive.google.com/file/d/1zKuWqm9weHSm98SZKld1PbEddgLOQkQV/view). And change 'PRETRAIN' in config/train_gnesf_3d.yaml.


### Train on ScanNet

```bash
sh scripts/train_gnesf_3d.sh
```
Remember to change the data path and pretrained model path in the shell script and ./config/train_gnesf_3d.yaml.


### Evaluate on ScanNet
```bash
sh scripts/eval_miou.sh
```
Remember to change the data path and pretrained model path.

