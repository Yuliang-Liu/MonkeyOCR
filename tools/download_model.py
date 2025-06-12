from argparse import ArgumentParser
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--type', '-t', type=str, default="huggingface") #modelscope
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(script_dir, "model_weight")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.type == "huggingface":
        from huggingface_hub import snapshot_download
        # 获取脚本所在目录的上级目录（MonkeyOCR根目录）
        snapshot_download(repo_id="echo840/MonkeyOCR", local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)
    elif args.type == "modelscope":
        from modelscope import snapshot_download as modelscope_download
        modelscope_download(repo_id = 'l1731396519/MonkeyOCR',local_dir=model_dir)
