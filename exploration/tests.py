from src.pipeline import Pipeline

cfg = {
    "path_to_train": "../datasets/downsample_train",
    "path_to_test": "../datasets/test",
    "batch_size": 32,
    "activation_type": "relu",
    "conv_pool_type": "ConvConvPool",
    "n_conv": 2,
    "n_epoch": 3,
    "archi_save_path": "./trained_model_tests/test_cfg_archi.json",
    "weights_save_path": "./trained_model_tests/test_cfg_weights.h5",
}

pipeline = Pipeline(**cfg)

pipeline.run_train()
breakpoint()
pipeline.run_test()
