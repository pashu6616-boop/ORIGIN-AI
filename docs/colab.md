# Google Colab T4 workflow

Use [`../notebooks/ORIGIN_AI_Colab.ipynb`](../notebooks/ORIGIN_AI_Colab.ipynb) for the complete notebook. In Colab, select a Tesla T4 GPU runtime, open the notebook, and run cells in order. It verifies CUDA and PyTorch, mounts Drive explicitly, installs SentencePiece/NumPy/PyArrow without replacing Colab's CUDA-enabled PyTorch, and asks for the project ZIP if `/content/ORIGIN-AI` is absent.

The notebook downloads WikiText-2 to `/content/drive/MyDrive/ORIGIN-AI/data/`, runs the repository smoke test before full training, and stores checkpoints, caches, and training/evaluation logs under `/content/drive/MyDrive/ORIGIN-AI/`. When rerun, it validates the saved checkpoint and tokenizer before adding the configured number of steps with `--resume`. The held-out validation split is used for evaluation; the test split is left untouched.

The checked-in configuration is intentionally small and experimental. A T4 makes the GPU path available but does not make the resulting model comparable to a pretrained general-purpose assistant. Colab may disconnect; rerunning the training cell resumes from a valid Drive checkpoint.
