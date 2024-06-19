# Fruit Object Detector

Welcome to the **Fruit Object Detector** repository! This repository demonstrates how to perform experiment tracking using [Weights & Biases (wandb)](https://wandb.ai/site) on an object detection model. The experiments and tests have been executed using a Colab Notebook to showcase the capabilities of the research.

## Colab Notebook

Explore the [Colab Notebook](https://colab.research.google.com/drive/1oU9ixj54Mryv4zAswIcJofD9Qryt9gR8?usp=sharing) which provides a step-by-step guide on setting up and running the object detection model with wandb experiment tracking.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [Weights & Biases](https://wandb.ai/site)

### Installation

1. Clone the YOLOv5 repository:
    ```bash
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    pip install wandb
    ```

3. Set up Weights & Biases:
    ```bash
    wandb login
    ```

### Dataset Preparation

Ensure your dataset is structured correctly with the images and annotations in YOLO format. The structure should be:

```
/dataset
    /images
        /train
        /val
    /labels
        /train
        /val
```

Create a `data.yaml` file to configure the dataset paths and class names:

```yaml
train: /path/to/dataset/images/train
val: /path/to/dataset/images/val

nc: 3  # number of classes
names: ['apple', 'banana', 'orange']  # class names
```

### Training the Model

Use the following script to train the YOLOv5 model with wandb integration:

```python
import wandb

def train_yolov5(epochs, batch_size, img_size, learning_rate, weight_decay):
    wandb.init(project='YOLOv5_Fruit_Detection', config={
        'epochs': epochs,
        'batch_size': batch_size,
        'img_size': img_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    })
    
    with open('hyp.yaml', 'w') as f:
        f.write(f'''
lr0: {learning_rate}
weight_decay: {weight_decay}
        ''')

    config = {
        'epochs': epochs,
        'batch_size': batch_size,
        'img_size': img_size,
        'data': 'data.yaml',
        'cfg': 'yolov5s.yaml',
        'weights': 'yolov5s.pt',
        'name': f'yolov5_fruit_{epochs}_{batch_size}_{img_size}_{learning_rate}_{weight_decay}',
        'project': 'YOLOv5_Fruit_Detection',
        'hyp': 'hyp.yaml'
    }

    !python train.py --img {config['img_size']} --batch {config['batch_size']} --epochs {config['epochs']} --data {config['data']} --cfg {config['cfg']} --weights {config['weights']} --hyp {config['hyp']} --project {config['project']} --name {config['name']}
    
    wandb.finish()

# Example usage
experiments = [
    {'epochs': 10, 'batch_size': 16, 'img_size': 640, 'learning_rate': 0.01, 'weight_decay': 0.0005},
    {'epochs': 20, 'batch_size': 16, 'img_size': 640, 'learning_rate': 0.01, 'weight_decay': 0.0005},
    {'epochs': 10, 'batch_size': 32, 'img_size': 640, 'learning_rate': 0.001, 'weight_decay': 0.0005},
    {'epochs': 10, 'batch_size': 16, 'img_size': 640, 'learning_rate': 0.01, 'weight_decay': 0.005},
]

for exp in experiments:
    train_yolov5(**exp)
```

### Results and Analysis

The results of each experiment, including loss graphs, evaluation metrics, example predictions, and more, will be automatically logged to your Weights & Biases project. You can analyze the importance of different hyperparameters and compare experiments directly within the wandb interface.

Visit your [wandb project page](https://wandb.ai) to explore the logged data and gain insights from the training runs.

## Contributing

We welcome contributions to improve the Fruit Object Detector. Please open issues or submit pull requests with any enhancements or bug fixes.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for providing a powerful and flexible object detection framework.
- [Weights & Biases](https://wandb.ai/site) for providing robust experiment tracking and visualization tools.
