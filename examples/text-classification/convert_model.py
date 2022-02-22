import argparse
import logging
import onnxruntime as ort

from pathlib import Path
from transformersX import convert_graph_to_onnx

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='PyTorch model dir')
    parser.add_argument('--onnx_dir', type=str, help='ONNX model dir')
    parser.add_argument('--task_name', type=str, help='Task name')
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    logger.info(f'convert model from {args.model_dir} into ONNX format')
    convert_graph_to_onnx.convert(
        framework="pt",
        model=args.model_dir,
        output=Path(args.onnx_dir),
        opset=12,
        tokenizer=args.model_dir,
        use_external_format=False,
        pipeline_name=args.task_name,
    )
    logger.info(f'save ONNX model as {args.onnx_dir}')

if __name__ == "__main__":
    main()
