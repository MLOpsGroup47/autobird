from contextlib import asynccontextmanager
from tkinter import Image

import hydra
from call_of_func.dataclasses.pathing import PathConfig
from call_of_func.utils.get_source_path import ROOT, CONFIG_DIR
import torch
from fastapi import FastAPI, File, UploadFile
from call_of_func.data.processing import preprocess_cfg
from call_of_birds_autobird.model import Model
from omegaconf import DictConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, feature_extractor, tokenizer, device, gen_kwargs
    print("Loading model")
    model = Model()
    model.load_state_dict(torch.load(PathConfig.ckpt_dir / "last.pth", map_location="cpu"))

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

    yield

    print("Cleaning up")
    del model, feature_extractor, tokenizer, device, gen_kwargs


app = FastAPI(lifespan=lifespan)


@app.post("/classify/")
async def caption(data: UploadFile = File(...)):

@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="configs")
def predict_step(cfg:DictConfig, filepath: str) -> str:
    print("Loading model")
    model = Model()
    model.load_state_dict(torch.load(PathConfig.ckpt_dir / "last.pth", map_location="cpu"))

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pr

    pred = "bird"
    
    return pred

if __name__ == "__main__":
      
    print(predict_step('s7_deployment/exercise_files/my_cat.jpg'))