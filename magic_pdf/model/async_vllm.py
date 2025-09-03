import os
import time
import asyncio
from typing import List
import torch
from loguru import logger
from magic_pdf.utils.load_image import load_image


class MonkeyChat_vLLM_async:
    def __init__(self, model_path, tp=1):
        try:
            from vllm import AsyncLLMEngine, SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs
        except ImportError:
            raise ImportError("vLLM is not installed. Please install it: "
                              "https://github.com/Yuliang-Liu/MonkeyOCR/blob/main/docs/install_cuda.md")

        self.model_name = os.path.basename(model_path)

        engine_args = AsyncEngineArgs(
            model=model_path,
            max_seq_len_to_capture=10240,
            mm_processor_kwargs={'use_fast': True},
            gpu_memory_utilization=self._auto_gpu_mem_ratio(0.9),
            disable_log_stats=True,
            enable_prefix_caching=True,
            tensor_parallel_size=tp
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"vLLM Async engine initialized: {self.model_name}")

        self.gen_config = SamplingParams(
            max_tokens=4096,
            temperature=0,
            repetition_penalty=1.05,
        )

        self.request_timeout = 600
    
    def _auto_gpu_mem_ratio(self, ratio):
        mem_free, mem_total = torch.cuda.mem_get_info()
        ratio = ratio * mem_free / mem_total
        return ratio

    async def async_batch_inference(self, images: List[str], questions: List[str]) -> List[str]:
        if len(images) != len(questions):
            raise ValueError("Images and questions must have the same length")

        semaphore = asyncio.Semaphore(min(64, max(1, len(images))))
        timeout_s = 300

        async def infer_one(img_path: str, q: str, req_id: str) -> str:
            placeholder = "<|image_pad|>"
            prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
                f"{q}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            inputs = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": [load_image(img_path, max_size=1600)],
                }
            }

            start = time.time()
            final_output = None
            async for out in self.engine.generate(inputs, self.gen_config, req_id):
                if time.time() - start > timeout_s:
                    try:
                        abort_res = self.engine.abort(req_id)
                        if asyncio.iscoroutine(abort_res):
                            await abort_res
                        logger.info(f"{req_id} aborted due to timeout")
                    except Exception as abort_err:
                        logger.warning(f"Abort failed for {req_id}: {abort_err}")
                    return "Error: Request timed out"
                final_output = out
                if getattr(out, "finished", False):
                    break

            if final_output and getattr(final_output, "outputs", None):
                return final_output.outputs[0].text
            return "Error: No output generated"

        async def bounded(img: str, q: str, idx: int):
            async with semaphore:
                req_id = f"batch_req_{idx}_{int(time.time()*1000)}"
                try:
                    return await infer_one(img, q, req_id)
                except Exception as e:
                    logger.error(f"Task {idx} failed: {e}")
                    return f"Error: {str(e)}"

        tasks = [bounded(img, q, i) for i, (img, q) in enumerate(zip(images, questions))]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"Task {i} exception: {r}")
                out.append(f"Error: {str(r)}")
            else:
                out.append(r)
        return out

    def batch_inference(self, images: List[str], questions: List[str]) -> List[str]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.async_batch_inference(images, questions))

        import concurrent.futures

        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(self.async_batch_inference(images, questions))
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(run_in_thread)
            try:
                return fut.result(timeout=self.request_timeout)
            except Exception as e:
                logger.error(f"Synchronous batch inference failed: {e}")
                return [f"Error: {str(e)}"] * len(images)