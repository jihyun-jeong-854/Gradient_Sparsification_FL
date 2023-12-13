# Gradient_Sparsification_FL
top k gradient sparsification in federated learning </br>

Dataset : MNIST, FEMNIST</br>
Model : Lenet, MobileNetv2, AlexNet</br>


## 1. Generate Dataset
You need to download FEMNIST from [https://github.com/TalwalkarLab/leaf](https://github.com/TalwalkarLab/leaf) </br>
It should includes the file 'images_by_writer.pkl'.

```
cd dataset
python generate_femnist.py --data_path {download_femnist_path} --num_clients
```
Generated Data path:  generated dataset/{datasetname}.

## 2. Train Model
- m : model,
- data : dataset
- nb : number of classes (mnist: 10, femnist: 62)
- nc : number of clients 
- tk : k (top k)
- tkalgo : the way to select top k gradients(global / chunk)

```
python main.py -m mobilenetv2 -d mnist -nb 10 -nc 100 -tk 100 -tkalgo global
```
