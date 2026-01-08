#!/usr/bin/env python3
import click
import os
import sys
from pathlib import Path
from loguru import logger

sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline import BidGenerator


@click.command()
@click.option('--input', required=True, help='招标文件路径（.docx）')
@click.option('--history', multiple=True, help='历史标书路径（可多选）')
@click.option('--output', default='output/final_bid.docx', help='输出文件路径')
def main(input, history, output):
    logger.info(f"招标文件: {input}")
    logger.info(f"历史标书: {list(history)}")
    
    generator = BidGenerator()
    
    history_list = list(history) if history else None
    
    try:
        result = generator.generate(input, history_list)
        logger.success(f"标书生成完成: {result['output_file']}")
    except Exception as e:
        logger.error(f"标书生成失败: {e}")
        raise


if __name__ == '__main__':
    main()
