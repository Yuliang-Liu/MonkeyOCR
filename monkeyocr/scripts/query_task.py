import sys
import argparse
from celery import Celery
from unobpi.main import celery_app

def query_result(task_id):
    async_result = celery_app.AsyncResult(task_id)
    print(f"Task [{task_id}] status: {async_result.status}")
    if async_result.status == "SUCCESS":
        print("Result:", async_result.result)
    elif async_result.status == "FAILURE":
        print("Error:", async_result.result)
    else:
        print("Task is still pending or running.")

def main():
    parser = argparse.ArgumentParser(description="查询 Celery 任务状态和结果")
    parser.add_argument("--task-id", type=str, required=True, help="要查询的 task_id")
    args = parser.parse_args()

    query_result(args.task_id)

if __name__ == "__main__":
    main() 