# Face Recognition

Este es un proyecto para la creación de un modelo de reconocimiento de spoofing utilizando TensorFlow y otras bibliotecas de Python.

# Training

````
docker build -t face-recognition-training -f DockerfileTrain .

docker run --rm -v "${PWD}/dataset:/app/data" -v "${PWD}/model:/app/model" face-recognition-training --data_dir "/app/data" --epochs 30 --batch_size 32 --output_model "/app/model/fasnet_trained.h5"
````