# Early Exit Neural Network

## Requirements
- Python 3.9+
- install required packages by `pip install -r requirements.txt`

## Usage
Support models
- ResNet (50, 101)
- VGG (11, 13, 16, 19)

You can defined your own model with the `BaseModel` class in `src/nn/model/base_model.py`. 

The custom model should have following properties before building the corresponding Early Exit Neural Network:
- `self.backbone`: The backbone layers of the model
- `self.classifier_module`: The classifier module of the model. Default is `ClassifierHead` in `src/nn/model/classifier.py`
- `self.classifier_config`: The input arguments for initialize the classifier module

Here is an example to define your own model:
```python
for EENN.src.nn.model.base_model import BaseModel

class YourModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super(YourModel, self).__init__(*args, **kwargs)
        # Define your model here
```

### Train regular model
An example to train a ResNet model on CIFAR 10, you can use the following command:
```bash
python train_model.py --dataset cifar --num_classes 10 --test_split test --model resnet50 --epoch 10 --cuda --verbose
```
To check more arguments, you can use `python train_model.py --help`

### Train Early Exit Neural Network
An example to train a ResNet model on CIFAR 10 with early exit, you can use the following command:
```bash
python train_eenn.py --dataset cifar --num_classes 10 --test_split test --model resnet50 --ee 3 5 7 9 11 13 15 17 19 --pretrained <path_to_pretrained_regular_model> --epoch 10 --cuda --verbose
```

Two examples to use the trained model for inference are provided in `example.py`.

### Profiling model
To profile the processing time of each layer in the model, and get the data size of the intermediate result, you can use the following command:
```bash
python model_profiling.py --model_path <model_path> --data_path <data_path> --save_folder <save_folder> --mode 0 1 2 3 4 5 6 7 --gate
```
- `data_path`: The path to the images

The result will be saved in `<save_folder/profile_summary.json>`

- `profile_summary['backbone_execution_time']['layer_name']`: processing time of each layer in backbone.
- `profile_summary['gate_execution_time']['layer_name']`: processing time of layers in each gate.
