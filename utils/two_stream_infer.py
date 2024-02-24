from models.two_stream_lipnet import TwoStreamLipNet
import options as opt
import os
import torch
import streamlit as st

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


@st.cache_resource
def load_model():
    model = TwoStreamLipNet()
    model = model.to(opt.device)

    # load the pretrained weights
    if hasattr(opt, "two_stream_weights"):
        pretrained_dict = torch.load(
            opt.two_stream_weights, map_location=torch.device(opt.device)
        )
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict.keys() and v.size() == model_dict[k].size()
        }
        missed_params = [
            k for k, v in model_dict.items() if not k in pretrained_dict.keys()
        ]
        print(
            "loaded params/tot params:{}/{}".format(
                len(pretrained_dict), len(model_dict)
            )
        )
        print("miss matched params:{}".format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model
