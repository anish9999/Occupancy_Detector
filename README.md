
# Occupany Tracker
Capacity trackering in real time in a bus. It is able to track the number of users with in a accumulated place.


## Running Tests

To run tests, run the following command

```bash
  python object_tracker_copy.py
```


## Documentation

1. [Pyenv](https://k0nze.dev/posts/install-pyenv-venv-vscode/) : constantly changing python version in a  devices
2. [Pyenv github link](https://github.com/pyenv-win/pyenv-win) : pyenv-win and version is necessary in this  folder
## Optimizations
For more optimize result , I considered yolov3 algorithm for training of data and considering different outcomes process for counting the passenger within an area


## Roadmap

- python version: 3.7.9
- cuda version: 3.10v
- cudnn version: 3.7.6 for cuda version 3.10v


## Installation
Save them to their weights folder

```bash
    https://pjreddie.com/media/files/yolov3.weights
    https://pjreddie.com/media/files/yolov3-tiny.weights
```
For checking if yolo's weights is loaded

yolov3
```bash
    python convert.py
```
yolov3-tiny
```bash
    python convert.py --weights ./weights/yolov3-tiny.weights --output ./weights/yolov3-tiny.tf
```
## 

Checking GPU
```python
import tensorflow as tf
print(tf.test.is_gpu_available(cuda_only =False,min_cuda_capability= None))
```


## Screenshots

![Capture1](https://raw.githubusercontent.com/anish9999/Occupancy_Tracker/master/Screenshot/Capture3.png?token=GHSAT0AAAAAAB4OQPJHBMGVWXNPIGGY6AGQY7GCVPQ)

![Capture2](https://raw.githubusercontent.com/anish9999/Occupancy_Tracker/master/Screenshot/Capture1.PNG?token=GHSAT0AAAAAAB4OQPJGSO5LTPNCTG5Q4USCY7GCXGQ)

![Capture3](https://raw.githubusercontent.com/anish9999/Occupancy_Tracker/master/Screenshot/Capture2.png?token=GHSAT0AAAAAAB4OQPJGYBMAQ37ELOVG3BQKY7GCUMA)
![Capture3](https://user-images.githubusercontent.com/85349550/218063737-54e590d8-2748-4b8c-a15f-28d6520810e2.png)


![Capture1](https://user-images.githubusercontent.com/85349550/218063633-d90f7310-2129-49bf-8f17-1fd9829ff705.png)
![Capture2](https://user-images.githubusercontent.com/85349550/218063702-21bdde59-41a7-465d-86fd-750c016d718d.png)


