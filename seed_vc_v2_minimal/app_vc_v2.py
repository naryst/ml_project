import gradio as gr
import torch
import yaml

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float16
def load_models(args):
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    cfg = DictConfig(yaml.safe_load(open(args.config, "r")))
    vc_wrapper = instantiate(cfg)
    vc_wrapper.load_checkpoints(ar_checkpoint_path=args.ar_checkpoint_path,
                                cfm_checkpoint_path=args.cfm_checkpoint_path)
    vc_wrapper.to(device)
    vc_wrapper.eval()

    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)

    if args.compile:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True

        if hasattr(torch._inductor.config, "fx_graph_cache"):
            # Experimental feature to reduce compilation times, will be on by default in future
            torch._inductor.config.fx_graph_cache = True
        vc_wrapper.compile_ar()
        # vc_wrapper.compile_cfm()

    return vc_wrapper

def main(args):
    vc_wrapper = load_models(args)
    
    # Set up Gradio interface
    description = ("Zero-shot voice conversion with in-context learning. For local deployment please check [GitHub repository](https://github.com/Plachtaa/seed-vc) "
                   "for details and updates.<br>Note that any reference audio will be forcefully clipped to 25s if beyond this length.<br> "
                   "If total duration of source and reference audio exceeds 30s, source audio will be processed in chunks.<br> ")
    
    inputs = [
        gr.Audio(type="filepath", label="Source Audio"),
        gr.Audio(type="filepath", label="Reference Audio"),
        gr.Slider(minimum=1, maximum=200, value=30, step=1, label="Diffusion Steps", 
                 info="30 by default, 50~100 for best quality"),
        gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Length Adjust", 
                 info="<1.0 for speed-up speech, >1.0 for slow-down speech"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.5, label="Intelligibility CFG Rate",
                 info="has subtle influence"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.5, label="Similarity CFG Rate",
                  info="has subtle influence"),
        gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.9, label="Top-p",
                 info="Controls diversity of generated audio"),
        gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Temperature",
                 info="Controls randomness of generated audio"),
        gr.Slider(minimum=1.0, maximum=3.0, step=0.1, value=1.0, label="Repetition Penalty",
                 info="Penalizes repetition in generated audio"),
        gr.Checkbox(label="convert style", value=False),
        gr.Checkbox(label="anonymization only", value=False),
    ]
    
    examples = [
        ["examples/source/yae_0.wav", "examples/reference/dingzhen_0.wav", 50, 1.0, 0.5, 0.5, 0.9, 1.0, 1.0, False, False],
        ["examples/source/jay_0.wav", "examples/reference/azuma_0.wav", 50, 1.0, 0.5, 0.5, 0.9, 1.0, 1.0, False, False],
    ]
    
    outputs = [
        gr.Audio(label="Stream Output Audio", streaming=True, format='mp3'),
        gr.Audio(label="Full Output Audio", streaming=False, format='wav')
    ]
    
    # Define a wrapper function to pass device and dtype
    def convert_voice_wrapper(*args):
        # Create a generator from the model's function
        generator = vc_wrapper.convert_voice_with_streaming(
            *args, 
            device=device,
            dtype=torch.float32  # Use float32 instead of float16
        )
        
        # For streaming output, we need to yield each chunk and return the final full audio
        stream_output = None
        full_output = None
        
        # Process the generator and get the last output
        for output in generator:
            stream_output, full_output = output
        
        # Return both the streaming and full outputs
        return stream_output, full_output
    
    # Launch the Gradio interface
    gr.Interface(
        fn=convert_voice_wrapper if device != "cuda" else vc_wrapper.convert_voice_with_streaming,
        description=description,
        inputs=inputs,
        outputs=outputs,
        title="Seed Voice Conversion V2",
        examples=examples,
        cache_examples=False,
    ).launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Compile the model using torch.compile")
    # V2 custom checkpoints
    parser.add_argument("--ar-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    parser.add_argument("--cfm-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    parser.add_argument('--config', type=str, default='configs/v2/vc_wrapper.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    main(args)