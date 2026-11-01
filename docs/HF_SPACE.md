# Hugging Face ZeroGPU demo

The Space in `hf_space/` is a standalone Gradio demo, separate from the local FastAPI application. It accepts MP4, MOV, AVI, and MKV videos up to 20 MB and 15 seconds. Output MP4 videos have no audio. The demo supports human review and is not a security decision-making system; predictions may be incorrect or miss detections.

Live Space: https://huggingface.co/spaces/luwukien/dangerous-person-detetion

## Preparing a deployment

- Review the provenance and distribution rights for the training data, checkpoint, dependencies, and sample videos. These rights have not been verified by this project. Only publish checkpoints and videos you have permission to distribute.
- The local checkpoint is `model-comparison/yolo11s-fit-within-640/weights/best.pt`, SHA-256 `be638b965b2978e8abcd5ef506e63ba19743fd88a0582d2a96ffdc390a9ba3f5`. Use a trusted checkpoint with class 0 `normal_person` and class 1 `potentially_dangerous_person`.
- Hosting on ZeroGPU requires an eligible Hugging Face account. Eligibility, queues, and quotas may change; consult the [ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu).

## Deploying or updating a Space

1. For a new Space, select the **Gradio SDK** and **ZeroGPU** hardware. Use a private Space during preparation if your account supports it.
2. Upload `README.md`, `app.py`, `video_processing.py`, and `requirements.txt` from `hf_space/` to the **Space repository root**. Do not upload tests or Python cache files. For an existing Space, upload only the changed files.
3. Once distribution rights are verified, upload the checkpoint as `best.pt` at the repository root. Do not upload datasets, tokens, or personal information. Set `MODEL_PATH` if using a different checkpoint path. The Space rebuilds automatically after a commit; check its logs if the build fails.
4. Test a short video you have permission to use. Verify playback and download, measure processing time, and confirm GPU operation in the logs. If GPU time or quota runs out, reduce video duration and retry. ZeroGPU may queue or reject requests when quota is exhausted.
5. Before making a new Space public, verify its permissions, content, and complete workflow. Include its URL and a backup screen recording in your README or CV.

Uploaded videos and results are stored temporarily on the server. Gradio periodically clears its cache; results older than one hour are removed on a subsequent request. Temporary files may remain after an unexpected process termination. Do not use sensitive videos. Long-term storage is not guaranteed.

## Local testing

Install `hf_space/requirements.txt` and the Gradio version specified in the Space metadata, then run `python hf_space/app.py`. Outside a Space, the demo uses CPU. The default checkpoint path is `hf_space/best.pt`; override it with `MODEL_PATH` as needed.

Run `python -m unittest discover -s hf_space/tests -v` to test video processing. ZeroGPU and its queue must be verified on the deployed Space.
