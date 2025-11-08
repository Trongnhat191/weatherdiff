## Tải mô hình
Có thể tải bằng wget

## Chỉnh file configs/mydataset.yml
Lúc load mô hình thì nó sẽ hiện số epochs và iters của mô hình pretrained đã được train
Sửa n_epochs và n_iters trong phần training sao cho lớn hơn số epochs và iters của mô hình pretrained

## Run train
python train_diffusion.py --config "mydataset.yml" --resume "WeatherDiff64.pth.tar"