# Occupany Tracker
Capacity trackering in real time in a bus. It is able to track the number of users with in a accumulated place.

# Demo
Click [Here](https://s9.gifyu.com/images/output_6.gif) to watch demo

## Main Concepts
![opencv(One project to another project diagram) drawio (5)](https://user-images.githubusercontent.com/85349550/219443098-57931f73-c5bf-4db9-8f81-0e4d7d08c4e9.png)

## Screenshots
![Capture3](https://user-images.githubusercontent.com/85349550/218063737-54e590d8-2748-4b8c-a15f-28d6520810e2.png)
![Capture2](https://user-images.githubusercontent.com/85349550/218063702-21bdde59-41a7-465d-86fd-750c016d718d.png)
![Capture1](https://user-images.githubusercontent.com/85349550/218063633-d90f7310-2129-49bf-8f17-1fd9829ff705.png)
![image](https://user-images.githubusercontent.com/85349550/223795168-63ab02ed-f9a1-4d33-a3b7-7eb8aceb8e3a.png)





## Dependencies

To run this project, you will need to add the following Dependencies to your file

`absl-py`
`numpy`
`seaborn`
`opencv-python`
`scipy`
`tensorflow`
`matplotlib`
`lxml`
`tqdm`

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
- GPU driver needed to get install for precision and accuracy


## Installations
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
![image](https://user-images.githubusercontent.com/85349550/223794918-c5ad3a9b-f820-4a55-a564-2d9027a6e6a6.png)
![image](https://user-images.githubusercontent.com/85349550/223794966-c0b2768c-96dc-4b70-b310-20b75bdf7790.png)
![image](https://user-images.githubusercontent.com/85349550/223795064-be2bcdb5-613b-44b7-98e9-698db784763a.png)
![image](https://user-images.githubusercontent.com/85349550/223795277-9150a2bc-e800-499f-95f2-85fb6453eefa.png)
![image](https://user-images.githubusercontent.com/85349550/223795390-81dddc26-20de-4f6d-b980-822db2408767.png)









