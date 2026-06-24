# ResNet50 backbone, ImageNet to its variants
# Before running the command, you need specify an evaluation output directory and the folder where the pretrianed mdoel is located
bash scripts/taskres/eval.sh imagenetv2 generalization_rn50 output/FINAL/dg/imagenetv2/seed1 output/FINAL/debug/imagenet/adapter_16shots/seed1/model
bash scripts/taskres/eval.sh imagenet_sketch generalization_rn50 output/FINAL/dg/imagenet_sketch/seed1 output/FINAL/debug/imagenet/adapter_16shots/seed1/model
bash scripts/taskres/eval.sh imagenet_a generalization_rn50 output/FINAL/dg/imagenet_a/seed1 output/FINAL/debug/imagenet/adapter_16shots/seed1/model
bash scripts/taskres/eval.sh imagenet_r generalization_50 output/FINAL/dg/imagenet_r/seed1 output/FINAL/debug/imagenet/adapter_16shots/seed1/model