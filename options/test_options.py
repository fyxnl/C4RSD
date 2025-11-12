from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        super().initialize()
        self.parser.add_argument("--test_res_dir", type=str, default="results")
        self.parser.add_argument("--test_dataset", type=str)
        self.parser.add_argument("--synthesis", action='store_true')
        self.parser.add_argument("--realworld", action='store_true')
        self.parser.add_argument("--model_name", type=str, default="Feng_last.pth")
        self.parser.add_argument("--pth_dir", type=str, default='results')
        self.isTrain = False
        