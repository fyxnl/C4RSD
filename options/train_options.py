from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        super().initialize()
        self.parser.add_argument("--train_dataset", type=str, default='Feng')
        self.parser.add_argument("--train_res_dir", type=str, default="results")
        self.parser.add_argument("--semi_supervised_pth", type=str)
        self.parser.add_argument("--supervised_pth", type=str)
        self.parser.add_argument("--img_size", type=int, default=256)
        self.parser.add_argument("--batch_size", type=int, default=10)
        self.parser.add_argument("--save_epoch", type=int, default=1)
        self.parser.add_argument("--beta1", type=float, default=0.9)
        self.parser.add_argument("--beta2", type=float, default=0.999)
        self.parser.add_argument("--lr", type=float, default=0.0001)
        self.parser.add_argument("--start_lr", type=float, default=0.0000001)
        self.parser.add_argument("--end_lr", type=float, default=  0.000000001)
        self.parser.add_argument("--epochs", type=int, default=20)
        self.isTrain = True
        