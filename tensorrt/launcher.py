import gradio as gr
import argparse
import os
from infer import infer


def infer_gradio(
    model, precision, index, warmup_iter, force_build_engine, progress=gr.Progress()
):
    # remove cached result before inference
    if os.path.exists("result.mp4"):
        os.remove("result.mp4")
    if os.path.exists("time.png"):
        os.remove("time.png")

    infer(
        model,
        precision,
        dataset_index=index,
        warmup_iter=warmup_iter,
        force_build_engine=force_build_engine,
        progress_logger=progress,
    )
    return "result.mp4", "time.png"


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--share",
        action="store_true",
        help="share the demo on internet, jetson(arm64) is not supported",
    )
    args = args.parse_args()

    demo = gr.Interface(
        fn=infer_gradio,
        inputs=[
            gr.Radio(
                ["sarnn", "cnnrnn", "cnnrnnln", "caebn"],
                label="Model name",
                value="sarnn",
            ),
            gr.Radio(
                ["fp32", "fp16", "int8"],
                label="Precision",
                value="fp32",
            ),
            gr.Slider(
                minimum=0,
                maximum=4,
                step=1,
                label="Index of the test data",
                value=0,
            ),
            gr.Number(
                value=1000,
                label="Warmup Iteration",
                precision=0,
                step=500,
                info="How many times to run the model before measuring performance.",
            ),
            gr.Checkbox(
                label="Force build engine",
                value=False,
                info="Ignore cached engine, and build new engine.",
            ),
        ],
        outputs=[gr.Video(), gr.Image()],
        title="EIPL web demo",
    )

    demo.launch(share=args.share)
