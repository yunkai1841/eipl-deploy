import gradio as gr
import argparse
from infer import infer


def infer_gradio(model, precision, index, warmup_iter, progress=gr.Progress()):
    infer(
        model,
        precision,
        dataset_index=index,
        warmup_iter=warmup_iter,
        progress_logger=progress,
    )
    return "result.mp4", "time.png"


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--share", action="store_true")
    args = args.parse_args()

    demo = gr.Interface(
        fn=infer_gradio,
        inputs=[
            gr.Radio(
                ["sarnn"],
                label="Model",
                value="sarnn",
                info="cnnrnn, cnnrnnln, caebn is not supported yet.",
            ),
            gr.Radio(
                # ["fp32", "fp16", "int8"],
                ["fp32"],
                label="Precision",
                value="fp32",
                info="fp16, int8 is not supported yet.",
            ),
            gr.Slider(
                minimum=0,
                maximum=100,
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
        ],
        outputs=[gr.Video(), gr.Image()],
        title="EIPL web demo",
    )

    demo.launch(share=args.share)
